import argparse
import json
from pathlib import Path
from time import time

import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file

from matrix_assembly import assemble_matrices, plot_assembly
from decode import beam_search
from sequence_assembly import simple_assembly, index2base # TODO: Rename assembly files
from model import get_prediction_model
from preprocess import mad_normalise, get_windows
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

    # args = parser.parse_args()
    # Local testing
    args = parser.parse_args(["data",
                              "data",
                              "--local",
                              "--rna-model=kmer_model/pc-transcripts-6mer-rna-model.json",
                              "--context-len=5"])

    # Local setup to avoid cuDNN error when running locally
    if args.local:
        setup_local()

    # read_to_resume = int(sys.argv[10])
    read_to_resume = 0

    # Load RNA model
    if args.rna_model != "None":
        with open(args.rna_model, "r") as f:
            rna_model_raw = json.load(f)
            # Format RNA model keys to format expected by beam search decoder
            rna_model = {}
            for context, dist in rna_model_raw.items():
                bases = ['A', 'C', 'G', 'T']
                context_formatted = tuple(map(lambda b: bases.index(b), context))
                rna_model[context_formatted] = dist
    else:
        print("No RNA model provided!")
        rna_model = None
    entropy_cache = {}

    # Load signal-to-sequence model
    sig_config = get_config(args.sig_config)
    sig_model = get_prediction_model(args.sig_model, sig_config)

    # Output to fastq
    fastq_n = 0
    fastq_i = 0
    fastq = open(f"{args.fasta_dir}/reads-{fastq_n}.fastq", "w")

    # Basecall each read in fast5 directory
    r = 0
    for fast5_filepath in Path(args.fast5_dir).rglob('*.fast5'):
        with get_fast5_file(fast5_filepath, 'r') as fast5:
            for read in fast5.get_reads():
                start_t = time()

                # Resume interrupted run
                if r < read_to_resume:
                    r += 1
                    continue

                # Preprocess read
                raw_signal = read.get_raw_data()
                try:
                    norm_signal = mad_normalise(raw_signal, args.outlier_clip)
                except ValueError as e:
                    print(e.args)
                    print(f"Signal preprocessing issue for {read.read_id}, skipping this read.")
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
                    # plot_assembly(matrices, matrix, args.chunk_len, args.step_size) # Debugging
                    sequence = beam_search(matrix,
                                           'ACGT',
                                           args.beam_width,
                                           rna_model,
                                           args.sig_threshold,
                                           args.rna_threshold,
                                           args.context_len,
                                           entropy_cache)
                elif args.decode_type == "local":
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
                else:
                    raise ValueError("Decoding type invalid")

                # TODO: Can we omit quality scores altogether without affecting
                # minimap2???
                # Create dummy Phred score
                dummy_phred = "+" * len(sequence)

                end_t = time()
                dur = end_t - start_t

                # Write read to fastq file (reverse sequence to be 5' to 3')
                fastq.write(f"@{read.read_id}\n{sequence[::-1]}\n+\n{dummy_phred}\n")
                fastq_i += 1
                print(f"[{r}] Basecalled read {read.read_id} in {dur:.2f} seconds")
                r += 1

                # Only write 100 reads per fastq file
                if fastq_i == 100:
                    fastq.close()
                    fastq_n += 1
                    fastq = open(f"{args.fasta_dir}/reads-{fastq_n}.fastq", "w")
                    fastq_i = 0

    # Make sure last fastq file is closed
    fastq.close()


if __name__ == "__main__":
    main()