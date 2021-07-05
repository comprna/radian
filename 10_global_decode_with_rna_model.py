import cProfile
import json
import os
from pathlib import Path

import numpy as np
from textdistance import levenshtein

from beam_search_decoder import ctcBeamSearch
from rna_model import get_rna_prediction_model
from utilities import get_config, setup_local

def main():
    # cProfile.run("callback()", sort="cumtime")

    # Set up RNA model

    setup_local()
    r_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/r-config-37.yaml'
    r_model_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/r-train-37-model-03.h5'
    r_config = get_config(r_config_file)
    r_model = get_rna_prediction_model(r_model_file, r_config)
    factor = 0.5

    # Data directory

    data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/8_TestGlobalDecoding/1_CombineSoftmaxes/hek293"

    # Load read IDs

    id_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/hek293/hek293_read_ids.npy"
    with open(id_file, "rb") as f:
        read_ids = np.load(f, allow_pickle=True)
    print(len(read_ids))

    # Load ground truth sequences

    gt_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/4_Construct_GT_Label_Per_Read/hek293_read_gt_labels.json"
    with open(gt_file, "r") as f:
        gts = json.load(f)

    # Combine all reads together so we can iterate more easily

    npy_paths = sorted(Path(data_dir).iterdir(), key=os.path.getmtime)
    with open(npy_paths[0], "rb") as f:
        global_softmaxes = np.load(f, allow_pickle=True)
    for path in npy_paths[1:]:
        with open(path, "rb") as f:
            global_softmaxes = np.concatenate((global_softmaxes, np.load(f, allow_pickle=True)))

    # Decode global softmax with RNA model

    classes = 'ACGT'
    for i, softmax in enumerate(global_softmaxes):
        print(f"GT:   {gts[read_ids[i]]}\n")
        pred_wout, _ = ctcBeamSearch(softmax, classes, None, None)
        print(f"Wout: {pred_wout}\n")
        pred_with, _ = ctcBeamSearch(softmax, classes, r_model, None, lm_factor=factor)
        print(f"With: {pred_with}\n")
        ed_with = levenshtein.normalized_distance(gts[read_ids[i]], pred_with)
        ed_wout = levenshtein.normalized_distance(gts[read_ids[i]], pred_wout)
        print(ed_with)
        print(ed_wout)
        if ed_with < ed_wout:
            print("BETTER\n\n")
        elif ed_with > ed_wout:
            print("WORSE\n\n")
        else:
            print("SAME\n\n")

# def callback():
#     npy_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/8_TestGlobalDecoding/1_CombineSoftmaxes/hek293/hek293_global_softmaxes_100.npy"

#     with open(npy_file, "rb") as f:
#         global_softmaxes = np.load(f, allow_pickle=True)

#     classes = 'ACGT'
#     ctcBeamSearch(global_softmaxes[0], classes, None, None)

if __name__ == "__main__":
    main()