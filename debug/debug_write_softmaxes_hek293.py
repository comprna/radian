import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K

from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()
    s_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/s-config-3.yaml'
    s_model_file = '/mnt/sda/rna-basecaller/experiments/sig-to-seq/dRNA/train-3/model-10.h5'

    s_config = get_config(s_config_file)
    s_model = get_prediction_model(s_model_file, s_config)

    # Read val data from numpy array
    
    data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/hek293"
    with open(f"{data_dir}/hek293_read_windows_all.npy", "rb") as f:
        windows_per_read = np.load(f, allow_pickle=True)

    # Load ground truth labels for each window

    gt_window_labels_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/hek293/hek293_read_labels_all.npy"
    with open(gt_window_labels_file, "rb") as f:
        gt_window_labels = np.load(f, allow_pickle=True)


    print(f"Number of reads: {len(windows_per_read)}")
    i = 0
    softmaxes_per_read = []
    for j, read in enumerate(windows_per_read):
        softmaxes = []
        for k, window in enumerate(read):
            softmax = s_model.predict(np.array([window,]))
            softmax = np.squeeze(softmax)
            softmaxes.append(softmax)

            print(gt_window_labels[j][k])
            plt.imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
            plt.show()

        print(i)
        i += 1
        softmaxes_per_read.append(softmaxes)

    # Write to file
    with open("softmaxes_per_read_hek293.npy", "wb") as f:
        np.save(f, softmaxes_per_read)


if __name__ == "__main__":
    main()
