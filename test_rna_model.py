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
    # Run locally or on gadi

    local_dir = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/copied_files'
    gadi_dir = '/g/data/xc17/Eyras/alex/working/with-rna-model/global/all_val/copied_files'

    base_dir = local_dir
    setup_local()

    # Load config

    r_config_file = f"{base_dir}/tmp/r-config-38.yaml"  # TODO: CL param
    r_config = get_config(r_config_file)

    d_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/decode-1/d-config-1.yaml' # TODO: CL param
    d_config = get_config(d_config_file)

    # Load RNA model

    r_model_file = f"{base_dir}/tmp/r-train-38-model-01.h5" # TODO: CL param
    r_model = get_rna_prediction_model(r_model_file, r_config)

    # Load read IDs

    id_file = f"{base_dir}/heart_read_ids.npy"
    with open(id_file, "rb") as f:
        read_ids = np.load(f, allow_pickle=True)
    print(len(read_ids))

    # Load ground truth sequences

    gt_file = f"{base_dir}/heart_read_gt_labels.json"
    with open(gt_file, "r") as f:
        gts = json.load(f)

    # Load global softmaxes

    gs_file = f"{base_dir}/heart_global_softmaxes_all.npy"
    with open(gs_file, "rb") as f:
        global_softmaxes = np.load(f, allow_pickle=True)

    # Decode global softmax with RNA model

    classes = 'ACGT'
    eds = []
    if d_config.use_rna_model == False:
        r_model = None
    for i, softmax in enumerate(global_softmaxes):
        pred, _ = ctcBeamSearch(softmax, classes, r_model, None, lm_factor=d_config.lm_factor)
        print(pred)
        ed = levenshtein.normalized_distance(gts[read_ids[i]], pred)
        eds.append(ed)
        print(f"{read_ids[i]}\t{gts[read_ids[i]]}\t{pred}\t{ed}")
    print(mean(eds))


if __name__ == "__main__":
    main()