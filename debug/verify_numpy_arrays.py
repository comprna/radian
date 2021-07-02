import json

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K

from beam_search_decoder import ctcBeamSearch
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()
    s_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/s-config-3.yaml'
    s_model_file = '/mnt/sda/rna-basecaller/experiments/sig-to-seq/dRNA/train-3/model-10.h5'

    s_config = get_config(s_config_file)
    s_model = get_prediction_model(s_model_file, s_config)

    data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/hek293"

    with open(f"{data_dir}/hek293_read_ids.npy", "rb") as f:
        read_ids = np.load(f, allow_pickle=True)

    with open(f"{data_dir}/hek293_read_labels_all.npy", "rb") as f:
        read_labels_all = np.load(f, allow_pickle=True)
    
    with open(f"{data_dir}/hek293_read_windows_all.npy", "rb") as f:
        read_windows_all = np.load(f, allow_pickle=True)

    print(len(read_ids))
    print(len(read_labels_all))
    print(len(read_windows_all))

    print(read_ids[0])
    print(len(read_labels_all[0]))
    print(len(read_windows_all[0]))

    # Do labels assemble to form GT label? PASS

    # gt_label_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/4_Construct_GT_Label_Per_Read/hek293_read_gt_labels.json"
    # with open(gt_label_file, "r") as f:
    #     gt_labels = json.load(f)

    # for i, read in enumerate(read_labels_all):
    #     print(gt_labels[read_ids[i]])
    #     for label in read:
    #         print(label)
    
    # Do labels match softmax?

    classes = 'ACGT'
    rna_model = None
    for i, read in enumerate(read_windows_all):
        if i < 10:
            continue
        for j, window in enumerate(read):
            softmax = s_model.predict(np.array([window,]))
            softmax = np.squeeze(softmax)

            pred = ctcBeamSearch(softmax, classes, rna_model, None)

            print(f"GT: {read_labels_all[i][j]}")
            print(f"PR: {pred}")

            plt.imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
            plt.show()



    


if __name__ == "__main__":
    main()