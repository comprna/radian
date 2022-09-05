from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file

from assembly import assemble_matrices, plot_assembly
from model import get_prediction_model
from preprocess import mad_normalise, get_windows
from utilities import get_config, setup_local


def main():
    setup_local()

    # This can contain single or multi fast5
    fast5_dir = "/mnt/sda/rna-basecaller/benchmarking/0_TestData/heart"

    # Parameters
    outlier_z_score = 4
    window_size = 1024
    step_size = 128
    batch_size = 32
    decode = "global"

    # Load signal-to-sequence model
    sig_config_file = '/mnt/sda/rna-basecaller/benchmarking/2_SigModel/s-config-3.yaml'
    sig_config = get_config(sig_config_file)
    sig_model_file = '/mnt/sda/rna-basecaller/benchmarking/2_SigModel/s-model-3-10.h5'
    sig_model = get_prediction_model(sig_model_file, sig_config)

    # Basecall each read in fast5 directory
    for fast5_filepath in Path(fast5_dir).rglob('*.fast5'):
        with get_fast5_file(fast5_filepath, 'r') as fast5:
            for read in fast5.get_reads():   
                # Preprocess read
                raw_signal = read.get_raw_data()
                norm_signal = mad_normalise(raw_signal, outlier_z_score)
                windows = get_windows(norm_signal, window_size, step_size)

                # Pass through signal-to-sequence model
                i = 0
                read_matrices = []
                while i + batch_size <= len(windows):
                    batch = windows[i:i+batch_size]
                    i += batch_size
                    read_matrices.append(sig_model.predict(batch))
                
                # Decode CTC output (with/without RNA model, global/local)

                if decode == "global":
                    matrix = assemble_matrices(read_matrices, step_size)
                    # plot_assembly(read_matrices, matrix, window_size, step_size) # Debugging
                    # ctc_decode(matrix)
                # else:
                #     for each window:
                #         ctc_decode(window)
                #     assemble_read_fragments

                # read_matrix = combine_matrices(all_matrices, step_size)

                # matrix_stack = stack_matrices(all_matrices, step_size)


                # Now that we have the stack, collapse it to get the final
                # global softmax so that there is only one distribution per
                # timestep.

                # Write to fastq


if __name__ == "__main__":
    main()