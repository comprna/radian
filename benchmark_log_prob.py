import json
import time

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import matplotlib.pyplot as plt
import numpy as np

from beam_search_decoder import ctcBeamSearch
from easy_assembler import simple_assembly, index2base
from utilities import setup_local

def main():
    # Run locally or on gadi

    setup_local()
    base_dir = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/copied_files'

    # Load config

    dataset = 'heart'

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

    # Decode  first read in heart dataset without RNA model

    r_model= None
    classes = 'ACGT'
    i = 0

    # Global decoding

    pred, _ = ctcBeamSearch(global_softmaxes[i],
                            classes,
                            r_model,
                            6,
                            0.5,
                            0.9,
                            8,
                            None)

    # Local decoding

    for i, softmax in enumerate(local_softmaxes[i]):
        pred, _ = ctcBeamSearch(softmax,
                                classes,
                                r_model,
                                6,
                                0.5,
                                0.9,
                                8,
                                None)


if __name__ == "__main__":
    main()
