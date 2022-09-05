import cProfile

import json

import numpy as np

from decode_beam_scoring import beam_search
from rna_model_lstm import get_rna_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()

    cProfile.run("callback()", sort="cumtime")

def callback():
    local_dir = base_dir = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/copied_files'

    # Load config

    dataset = "heart"

    # Load RNA model

    r_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/local/train-3-37/r-config-37.yaml'
    r_config = get_config(r_config_file)
    r_model_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/local/train-3-37/r-train-37-model-03.h5'
    r_model = get_rna_prediction_model(r_model_file, r_config)

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
    beam_width = 30
    lm_factor = 0.5
    len_context = 8
    cache = {}

    # Predict with model & with thresholds
    beam_search(global_softmaxes[0],
                bases,
                beam_width,
                r_model,
                lm_factor,
                len_context,
                cache)


if __name__ == "__main__":
    main()