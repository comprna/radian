import json
from statistics import mean
import sys

import numpy as np
from textdistance import levenshtein

from decode_beam_scoring import beam_search
from rna_model_lstm import get_rna_prediction_model
from utilities import get_config, setup_local

def main():
    # Run locally or on gadi

    setup_local()
    base_dir = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/copied_files'
    dataset = "heart"

    # Load config

    len_context = 8
    beam_width = 30
    lm_factor = 0.5
    normalise_after = True


    # Load RNA model

    r_model_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/local/train-3-37/r-train-37-model-03.h5'
    if r_model_file == "None":
        r_model = None
    else:
        r_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/local/train-3-37/r-config-37.yaml'
        r_config = get_config(r_config_file)
        r_model = get_rna_prediction_model(r_model_file, r_config, len_context)

    print(f"{r_model}\t{type(r_model)}")
    print(f"{len_context}\t{type(len_context)}")
    print(f"{beam_width}\t{type(beam_width)}")
    print(f"{lm_factor}\t{type(lm_factor)}")
    print(f"{normalise_after}\t{type(normalise_after)}")

    # Load read IDs

    id_file = f"{base_dir}/{dataset}_read_ids.npy"
    with open(id_file, "rb") as f:
        read_ids = np.load(f, allow_pickle=True)

    # Load global ground truth sequences

    gt_file = f"{base_dir}/{dataset}_read_gt_labels.json"
    with open(gt_file, "r") as f:
        global_gts = json.load(f)

    # Load global softmaxes

    gs_file = f"{base_dir}/{dataset}_global_softmaxes_all.npy"
    with open(gs_file, "rb") as f:
        global_softmaxes = np.load(f, allow_pickle=True)

    # Decode global softmax with RNA model

    bases = 'ACGT'
    cache = {}

    for i, softmax in enumerate(global_softmaxes):
        if i % 4 != 0:
            continue

        if i == 0:
            continue

        # Ground truth
        gt = global_gts[read_ids[i]]
        print(gt)

        # Predict with model & with thresholds
        pred, _ = beam_search(softmax,
                              bases,
                              beam_width,
                              r_model,
                              lm_factor,
                              len_context,
                              cache,
                              normalise_after)
        ed = levenshtein.normalized_distance(gt, pred)
        
        print(f"{i}\t{read_ids[i]}\t{gt}\t{pred}\t{ed}")



if __name__ == "__main__":
    main()