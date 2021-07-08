import cProfile
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
from textdistance import levenshtein

from beam_search_decoder import ctcBeamSearch
from rna_model import get_rna_prediction_model
from utilities import get_config, setup_local

def main():
    # Set up RNA model

    gadi_dir = '/g/data/xc17/Eyras/alex/working/test_decoding'
    r_config_file = f"{gadi_dir}/r-config-38.yaml"
    r_model_file = f"{gadi_dir}/r-train-38-model-01.h5"
    r_config = get_config(r_config_file)
    r_model = get_rna_prediction_model(r_model_file, r_config)
    factor = 0.5

    # Load read IDs

    id_file = f"{gadi_dir}/heart_read_ids.npy"
    with open(id_file, "rb") as f:
        read_ids = np.load(f, allow_pickle=True)
    print(len(read_ids))

    # Load ground truth sequences

    gt_file = f"{gadi_dir}/heart_read_gt_labels.json"
    with open(gt_file, "r") as f:
        gts = json.load(f)

    # Load global softmaxes

    gs_file = f"{gadi_dir}/heart_global_softmaxes_all.npy"
    with open(gs_file, "r") as f:
        global_softmaxes = np.load(f, allow_pickle=True)

    # Decode global softmax with RNA model

    classes = 'ACGT'
    eds = []
    for i, softmax in enumerate(global_softmaxes):
        # pred, _ = ctcBeamSearch(softmax, classes, None, None)
        pred, _ = ctcBeamSearch(softmax, classes, r_model, None, lm_factor=factor)
        ed = levenshtein.normalized_distance(gts[read_ids[i]], pred)
        eds.append(ed)
        print(f"{read_ids[i]}\t{gts[read_ids[i]]}\t{pred}\t{ed}")
    print(mean(eds))


if __name__ == "__main__":
    main()