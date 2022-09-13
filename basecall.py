import json
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file

from matrix_assembly import assemble_matrices, plot_assembly
from decode_dynamic import beam_search
from sequence_assembly import simple_assembly, index2base # TODO: Rename assembly files
from model import get_prediction_model
from preprocess import mad_normalise, get_windows
from utilities import get_config, setup_local


def main():
    # Comment out if running on gadi
    setup_local()

    # Data directories
    fast5_dir = "/home/alex/Documents/tmp/fast5" # Single or multi fast5s
    fastq_dir = 'fastq'
    # fast5_dir = sys.argv[1]
    # fastq_dir = sys.argv[2]

    # Preprocessing parameters
    outlier_z_score = 4
    window_size = 1024
    step_size = 128
    batch_size = 32

    # Model files
    rna_model_file = "kmer_model/transcripts-6mer-rna-model.json"
    sig_config_file = '/mnt/sda/rna-basecaller/benchmarking/2_SigModel/s-config-3.yaml'
    sig_model_file = '/mnt/sda/rna-basecaller/benchmarking/2_SigModel/s-model-3-10.h5'
    # rna_model_file = sys.argv[3]
    # sig_config_file = sys.argv[4]
    # sig_model_file = sys.argv[5]

    # Decoding parameters
    beam_width = 6
    decode = "global"
    s_threshold = 0.5
    r_threshold = 0.5
    context_len = 5
    # decode = sys.argv[6]
    # s_threshold = float(sys.argv[7])
    # r_threshold = float(sys.argv[8])
    # context_len = int(sys.argv[9])

    # Load RNA model
    if rna_model_file != "None":
        with open(rna_model_file, "r") as f:
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
    sig_config = get_config(sig_config_file)
    sig_model = get_prediction_model(sig_model_file, sig_config)

    # Output to fastq
    fastq_n = 0
    fastq_i = 0
    fastq = open(f"{fastq_dir}/reads-{fastq_n}.fastq", "w")

    # Basecall each read in fast5 directory
    for fast5_filepath in Path(fast5_dir).rglob('*.fast5'):
        with get_fast5_file(fast5_filepath, 'r') as fast5:
            for read in fast5.get_reads(): 
                # Preprocess read
                raw_signal = read.get_raw_data()
                norm_signal = mad_normalise(raw_signal, outlier_z_score)
                windows, pad = get_windows(norm_signal, window_size, step_size)

                # Pass windows through signal model in batches
                i = 0
                matrices = []
                while i + batch_size <= len(windows):
                    batch = windows[i:i+batch_size]
                    i += batch_size
                    matrices.extend(sig_model.predict(batch))
                matrices.extend(sig_model.predict(windows[i:]))

                # Trim padding from last matrix before decoding
                matrices[-1] = matrices[-1][:-pad]

                # Decode CTC output (with/without RNA model, global/local)
                if decode == "global":
                    matrix = assemble_matrices(matrices, step_size)
                    # plot_assembly(matrices, matrix, window_size, step_size) # Debugging
                    sequence = beam_search(matrix,
                                           'ACGT',
                                           beam_width, 
                                           rna_model,
                                           s_threshold,
                                           r_threshold,
                                           context_len,
                                           entropy_cache)
                elif decode == "local":
                    read_fragments = []
                    for matrix in matrices:
                        sequence = beam_search(matrix,
                                                'ACGT',
                                                beam_width,
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

                print(f"@{read.read_id}\n{sequence[::-1]}\n+\n{dummy_phred}\n")
                exit()

                # Write read to fastq file (reverse sequence to be 5' to 3')
                fastq.write(f"@{read.read_id}\n{sequence[::-1]}\n+\n{dummy_phred}\n")
                fastq_i += 1

                # Only write 1,000 reads per fastq file
                if fastq_i == 1000:
                    fastq.close()
                    fastq_n += 1
                    fastq = open(f"reads-{fastq_n}.fastq", "w")
                    fastq_i = 0

    # Make sure last fastq file is closed
    fastq.close()


if __name__ == "__main__":
    main()