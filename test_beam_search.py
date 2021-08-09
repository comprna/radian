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

    # Load local ground truth sequences

    gt_heart_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/heart/heart_read_labels_all.npy"
    with open(gt_heart_file, "rb") as f:
        local_gts = np.load(f, allow_pickle=True)

    # Load local softmaxes

    softmaxes_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/6_WriteSoftmaxes/softmaxes_per_read_heart.npy"
    with open(softmaxes_file, "rb") as f:
        local_softmaxes = np.load(f, allow_pickle=True)

    # Load global ground truth sequences

    gt_file = f"{base_dir}/{dataset}_read_gt_labels.json"
    with open(gt_file, "r") as f:
        global_gts = json.load(f)

    # Load global softmaxes

    gs_file = f"{base_dir}/{dataset}_global_softmaxes_all.npy"
    with open(gs_file, "rb") as f:
        global_softmaxes = np.load(f, allow_pickle=True)

    # Decode global softmax with RNA model

    classes = 'ACGT'
    beam_width = 6
    lm_factor = 0.5
    s_threshold = 0.6
    r_threshold = 10
    len_context = 8
    cache = {}

    decode_type = 'global'

    if decode_type == 'global':
        for i, softmax in enumerate(global_softmaxes):
            # Ground truth
            gt = global_gts[read_ids[i]]
            print(gt)

            # Predict without model
            # pred_a, _ = beam_search(softmax,
            #                         classes,
            #                         beam_width,
            #                         None,
            #                         None,
            #                         s_threshold,
            #                         r_threshold,
            #                         len_context,
            #                         cache)
            # ed_a = levenshtein.normalized_distance(gt, pred_a)
            # print(pred_a)
            # print(ed_a)

            # Predict with model & with thresholds
            pred_c, _ = beam_search(softmax,
                                    classes,
                                    beam_width,
                                    r_model,
                                    None,
                                    s_threshold,
                                    r_threshold,
                                    len_context,
                                    cache)
            ed_c = levenshtein.normalized_distance(gt, pred_c)
            print(ed_c)
    else:
        for i, read in enumerate(local_softmaxes):
            for j, softmax in enumerate(read):
                # Ground truth
                gt = local_gts[i][j]
                print(gt)

                # Predict without model
                pred_a, _ = beam_search(softmax,
                                        classes,
                                        beam_width,
                                        None,
                                        None,
                                        s_threshold,
                                        r_threshold,
                                        len_context,
                                        cache)
                ed_a = levenshtein.normalized_distance(gt, pred_a)
                print(pred_a)
                print(ed_a)

                # Predict with model & with thresholds
                pred_c, _ = beam_search(softmax,
                                        classes,
                                        beam_width,
                                        r_model,
                                        None,
                                        s_threshold,
                                        r_threshold,
                                        len_context,
                                        cache)
                ed_c = levenshtein.normalized_distance(gt, pred_c)
                print(pred_c)
                print(ed_c)
                print("\n\n")



if __name__ == "__main__":
    main()