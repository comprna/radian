import argparse
import json
from pathlib import Path
from time import time

import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
import pyslow5

from decode import beam_search
from matrix_assembly import assemble_matrices, plot_assembly
from model import get_prediction_model
from preprocess import mad_normalise, get_windows
from sequence_assembly import simple_assembly, index2base # TODO: Rename assembly files
from utilities import get_config, setup_local


def main():
    # CL args
    parser = argparse.ArgumentParser(description=("Basecall a nanopore dRNA "
                                                  "sequencing run."))
    parser.add_argument("fast5_dir", help="Directory of single/multi fast5 files.")
    parser.add_argument("fasta_dir", help="Directory to output fasta files.")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--chunk-len", default=1024, type=int)
    parser.add_argument("--step-size", default=128, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--outlier-clip", default=4, type=int)
    parser.add_argument("--rna-model", default="models/rnamodel_12mer_pc.json")
    parser.add_argument("--sig-model", default="models/sig2seq.h5")
    parser.add_argument("--sig-config", default="models/sig2seq.yaml")
    parser.add_argument("--beam-width", default=6, type=int)
    parser.add_argument("--decode-type", choices=["global", "chunk"], default="global")
    parser.add_argument("--sig-threshold", default=0.5, type=float)
    parser.add_argument("--rna-threshold", default=0.5, type=float)
    parser.add_argument("--context-len", default=11, type=int)

    args = parser.parse_args()
    # Local testing
    # args = parser.parse_args(["data",
    #                           "data",
    #                           "--local"])

    # Local setup to avoid cuDNN error when running locally
    if args.local:
        setup_local()

    # Load RNA model
    rna_model = args.rna_model
    if rna_model != "None":
        with open(args.rna_model, "r") as f:
            rna_model_raw = json.load(f)
            # Format RNA model keys to format expected by beam search decoder
            rna_model = {}
            for context, dist in rna_model_raw.items():
                bases = ['A', 'C', 'G', 'T']
                context_formatted = tuple(map(lambda b: bases.index(b), context))
                rna_model[context_formatted] = dist
    entropy_cache = {}

    # Load signal-to-sequence model
    sig_config = get_config(args.sig_config)
    sig_model = get_prediction_model(args.sig_model, sig_config)

    # Output to fasta
    fasta_n = 0
    fasta_i = 0
    fasta = open(f"{args.fasta_dir}/reads-{fasta_n}.fasta", "w")

    # Basecall each read in fast5 directory
    # this is a hack, just detecting the .blow5 extention to hijack the arg
    # should do something better than this
    if args.fast5_dir.split(".")[-1] == "blow5":
        SLOW5 = True
        s5 = pyslow5.Open(args.fast5_dir, 'r')
        # this can be multithreaded with s5.seq_reads_multi(), see docs for flags
        reads = s5.seq_reads()
        # this is copy/paste of below but using read_id set first rather than accessing it
        for read in reads:
            start_t = time()
            read_id = read["read_id"]
            # Preprocess read
            raw_signal = read["signal"]
            try:
                norm_signal = mad_normalise(raw_signal, args.outlier_clip)
            except ValueError as e:
                print(e.args)
                print(f"{read_id} signal issue, skipping this read.")
                continue
            windows, pad = get_windows(norm_signal, args.chunk_len, args.step_size)

            # Pass windows through signal model in batches
            i = 0
            matrices = []
            while i + args.batch_size <= len(windows):
                batch = windows[i:i+args.batch_size]
                i += args.batch_size
                matrices.extend(sig_model.predict(batch))
            if i < len(windows):
                matrices.extend(sig_model.predict(windows[i:]))

            # Trim padding from last matrix before decoding
            matrices[-1] = matrices[-1][:-pad]

            # Decode CTC output (with/without RNA model, global/local)
            if args.decode_type == "global":
                matrix = assemble_matrices(matrices, args.step_size)
                # plot_assembly(matrices, matrix, args.chunk_len, args.step_size)
                sequence = beam_search(matrix,
                                        'ACGT',
                                        args.beam_width,
                                        rna_model,
                                        args.sig_threshold,
                                        args.rna_threshold,
                                        args.context_len,
                                        entropy_cache)
            else:
                read_fragments = []
                for matrix in matrices:
                    sequence = beam_search(matrix,
                                            'ACGT',
                                            args.beam_width,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None)
                    read_fragments.append(sequence)
                consensus = simple_assembly(read_fragments)
                sequence = index2base(np.argmax(consensus, axis=0))

            end_t = time()
            dur = end_t - start_t

            # Write read to fasta file (reverse sequence to be 5' to 3')
            fasta.write(f">{read_id}\n{sequence[::-1]}\n")
            fasta_i += 1
            print(f"Basecalled read {read_id} in {dur:.2f} sec.")

            # Only write 1000 reads per fasta file
            if fasta_i == 1000:
                fasta.close()
                fasta_n += 1
                fasta = open(f"{args.fasta_dir}/reads-{fasta_n}.fasta", "w")
                fasta_i = 0
    else:
        for fast5_filepath in Path(args.fast5_dir).rglob('*.fast5'):
            with get_fast5_file(fast5_filepath, 'r') as fast5:
                for read in fast5.get_reads():
                    start_t = time()
                    
                    # Preprocess read
                    raw_signal = read.get_raw_data()
                    try:
                        norm_signal = mad_normalise(raw_signal, args.outlier_clip)
                    except ValueError as e:
                        print(e.args)
                        print(f"{read.read_id} signal issue, skipping this read.")
                        continue
                    windows, pad = get_windows(norm_signal, args.chunk_len, args.step_size)

                    # Pass windows through signal model in batches
                    i = 0
                    matrices = []
                    while i + args.batch_size <= len(windows):
                        batch = windows[i:i+args.batch_size]
                        i += args.batch_size
                        matrices.extend(sig_model.predict(batch))
                    if i < len(windows):
                        matrices.extend(sig_model.predict(windows[i:]))

                    # Trim padding from last matrix before decoding
                    matrices[-1] = matrices[-1][:-pad]

                    # Decode CTC output (with/without RNA model, global/local)
                    if args.decode_type == "global":
                        matrix = assemble_matrices(matrices, args.step_size)
                        # plot_assembly(matrices, matrix, args.chunk_len, args.step_size)
                        sequence = beam_search(matrix,
                                            'ACGT',
                                            args.beam_width,
                                            rna_model,
                                            args.sig_threshold,
                                            args.rna_threshold,
                                            args.context_len,
                                            entropy_cache)
                    else:
                        read_fragments = []
                        for matrix in matrices:
                            sequence = beam_search(matrix,
                                                    'ACGT',
                                                    args.beam_width,
                                                    None,
                                                    None,
                                                    None,
                                                    None,
                                                    None)
                            read_fragments.append(sequence)
                        consensus = simple_assembly(read_fragments)
                        sequence = index2base(np.argmax(consensus, axis=0))

                    end_t = time()
                    dur = end_t - start_t

                    # Write read to fasta file (reverse sequence to be 5' to 3')
                    fasta.write(f">{read.read_id}\n{sequence[::-1]}\n")
                    fasta_i += 1
                    print(f"Basecalled read {read.read_id} in {dur:.2f} sec.")

                    # Only write 1000 reads per fasta file
                    if fasta_i == 1000:
                        fasta.close()
                        fasta_n += 1
                        fasta = open(f"{args.fasta_dir}/reads-{fasta_n}.fasta", "w")
                        fasta_i = 0

    # Make sure last fasta file is closed
    fasta.close()


if __name__ == "__main__":
    main()