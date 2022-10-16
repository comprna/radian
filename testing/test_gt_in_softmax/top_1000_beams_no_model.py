import json

import numpy as np

from beam_search_decoder import beam_search
from testing.obsolete.rna_model_lstm import get_rna_prediction_model
from textdistance import levenshtein
from utilities import get_config, setup_local

def main():
    # Run locally or on gadi

    # setup_local()
    # base_dir = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/copied_files'
    base_dir = '/g/data/xc17/Eyras/alex/working/with-rna-model/global/all_val/copied_files'

    # Load config

    dataset = 'heart'

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

    # Set up RNA model

    # r_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/copied_files/tmp/r-config-37.yaml'
    # r_config = get_config(r_config_file)
    # r_model_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/copied_files/tmp/r-train-37-model-03.h5'
    # r_model = get_rna_prediction_model(r_model_file, r_config)

    # Decoding params

    beam_width = 1000
    factor = 0.5
    threshold = 0.9
    len_context = 8

    # Global decoding WITHOUT RNA model

    bases = 'ACGT'

    for i, _ in enumerate(read_ids):

        # Only analyse first 100 reads

        if i == 100:
            break

        top_beams = beam_search(global_softmaxes[i],
                                bases,
                                beam_width,
                                None,
                                factor,
                                threshold,
                                len_context)

        # Write top beams to file

        with open(f"{read_ids[i]}_top_1000_beams_no_model.tsv", "w") as f:
            for k, beam in enumerate(top_beams):
                beam_seq = ''.join([bases[label] for label in beam])
                ed = levenshtein.normalized_distance(global_gts[read_ids[i]], beam_seq)
                f.write(f"{read_ids[i]}\t{k}\t{global_gts[read_ids[i]]}\t{beam_seq}\t{ed}")



if __name__ == "__main__":
    main()
