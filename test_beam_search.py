import json
from statistics import mean
import sys

import numpy as np
from textdistance import levenshtein

from beam_search_decoder import beam_search
from rna_model import get_rna_prediction_model
from utilities import get_config, setup_local

def main():
    # Run locally or on gadi

    setup_local()
    local_dir = base_dir = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/copied_files'
    # gadi_dir = '/g/data/xc17/Eyras/alex/working/with-rna-model/global/all_val/copied_files'

    # base_dir = sys.argv[1]

    # Load config

    dataset = "heart"
    # d_config_file = sys.argv[3]
    # d_config = get_config(d_config_file)

    # Load RNA model

    r_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/local/train-3-37/r-config-37.yaml'
    r_config = get_config(r_config_file)
    r_model_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/local/train-3-37/r-train-37-model-03.h5'
    r_model = get_rna_prediction_model(r_model_file, r_config)

    # Load read IDs

    id_file = f"{base_dir}/{dataset}_read_ids.npy"
    with open(id_file, "rb") as f:
        read_ids = np.load(f, allow_pickle=True)

    # Load ground truth sequences

    gt_file = f"{base_dir}/{dataset}_read_gt_labels.json"
    with open(gt_file, "r") as f:
        gts = json.load(f)

    # Load global softmaxes

    gs_file = f"{base_dir}/{dataset}_global_softmaxes_all.npy"
    with open(gs_file, "rb") as f:
        global_softmaxes = np.load(f, allow_pickle=True)

    # Decode global softmax with RNA model

    classes = 'ACGT'
    beam_width = 30
    lm_factor = 0.5
    s_threshold = 0.6
    r_threshold = 0.6
    len_context = 8
    for i, softmax in enumerate(global_softmaxes):
        if i == 0:
            continue
        # Predict without model
        pred_a, _ = beam_search(softmax,
                                classes,
                                beam_width,
                                None,
                                None,
                                s_threshold,
                                r_threshold,
                                len_context)
        ed_a = levenshtein.normalized_distance(gts[read_ids[i]], pred_a)
        print(ed_a)

        # Predict with model & no thresholds
        pred_b, _ = beam_search(softmax,
                                classes,
                                beam_width,
                                r_model,
                                None,
                                float("-inf"),
                                float("+inf"),
                                len_context)
        ed_b = levenshtein.normalized_distance(gts[read_ids[i]], pred_b)
        print(ed_b)

        # Predict with model & with thresholds
        pred_c, _ = beam_search(softmax,
                                classes,
                                beam_width,
                                r_model,
                                None,
                                s_threshold,
                                r_threshold,
                                len_context)
        ed_c = levenshtein.normalized_distance(gts[read_ids[i]], pred_c)
        print(ed_c)



if __name__ == "__main__":
    main()