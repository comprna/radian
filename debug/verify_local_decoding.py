import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def main():
    ###################################################################
    # # Verify softmaxes per read match those used in global decoding - PASS

    # # Softmaxes used in local decoding

    # local_softmaxes_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/7_TestLocalDecoding/1_DecodeWindowSoftmaxes/hek293/gadi/copied_files/softmaxes_per_read_hek293.npy"
    # with open(local_softmaxes_file, "rb") as f:
    #     read_softmaxes_local = np.load(f, allow_pickle=True)

    # # Softmaxes used in global decoding

    # global_softmaxes_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/6_WriteSoftmaxes/softmaxes_per_read_hek293.npy"
    # with open(global_softmaxes_file, "rb") as f:
    #     read_softmaxes_global = np.load(f, allow_pickle=True)
    
    # # Check that softmaxes per read are same for both local and global
    # # decoding

    # for i,read in enumerate(read_softmaxes_local):
    #     print(len(read_softmaxes_local[i]))
    #     print(len(read_softmaxes_global[i]))

    #     for j, softmax in enumerate(read):
    #         fig, axes = plt.subplots(2, 1, sharex=True)
    #         axes[0].imshow(np.transpose(read_softmaxes_local[i][j]), cmap="gray_r", aspect="auto")
    #         axes[1].imshow(np.transpose(read_softmaxes_global[i][j]), cmap="gray_r", aspect="auto")
    #         plt.show()
    
    ###################################################################
    # # Verify local decoding is reasonable - FAILS (GT and preds seem to be from different reads)

    # # Load ground truth labels for each window

    # gt_window_labels_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/hek293/hek293_read_labels_all.npy"
    # with open(gt_window_labels_file, "rb") as f:
    #     gt_window_labels = np.load(f, allow_pickle=True)

    # # Combine all reads together so we can iterate more easily

    # data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/7_TestLocalDecoding/1_DecodeWindowSoftmaxes/hek293/numpy"
    # npy_paths = sorted(Path(data_dir).iterdir(), key=os.path.getmtime)
    # with open(npy_paths[0], "rb") as f:
    #     read_local_preds = np.load(f, allow_pickle=True)
    # for path in npy_paths[1:]:
    #     with open(path, "rb") as f:
    #         read_local_preds = np.concatenate((read_local_preds, np.load(f, allow_pickle=True)))
    # print(len(read_local_preds))

    # # Check each read's window labels

    # for i, read in enumerate(read_local_preds):
    #     for j, pred in enumerate(read):
    #         print(pred)
    #         print(gt_window_labels[i][j])

    ###################################################################
    # Verify window labels (in numpy file) match softmaxes per read

    # Load ground truth labels for each window

    gt_window_labels_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/hek293/hek293_read_labels_all.npy"
    with open(gt_window_labels_file, "rb") as f:
        gt_window_labels = np.load(f, allow_pickle=True)
    
    # Load softmaxes per read

    softmaxes_per_read_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/6_WriteSoftmaxes/softmaxes_per_read_hek293.npy"
    with open(softmaxes_per_read_file, "rb") as f:
        softmaxes_per_read = np.load(f, allow_pickle=True)

    # Verify that softmaxes match up with labels

    print(len(gt_window_labels))
    print(len(softmaxes_per_read))

    for i, read in enumerate(softmaxes_per_read):
        if i == 0:
            continue
        print(len(gt_window_labels[i]))
        print(len(softmaxes_per_read[i]))

        for j, softmax in enumerate(read):
            print(gt_window_labels[i][j])
            plt.imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
            plt.show()




if __name__ == "__main__":
    main()