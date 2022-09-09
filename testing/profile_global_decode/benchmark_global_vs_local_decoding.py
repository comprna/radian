import json
import time

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import matplotlib.pyplot as plt
import numpy as np

from beam_search_decoder import ctcBeamSearch
from sequence_assembly import simple_assembly, index2base
from utilities import setup_local

def overlay_prediction(plot, prediction, indices, x_min, x_max, color, offset=0):
    bases = ['A', 'C', 'G', 'T']
    texts = []
    for i, nt in enumerate(prediction):
        # Only annotate within the bounds of the figure
        if indices[i] >= x_min and indices[i] <= x_max:
            text = plot.text(indices[i], bases.index(nt)+offset, nt,  fontsize='xx-large', color=color, weight="bold")
            texts.append(text)
    return texts

def clear_texts(texts):
    for text in texts:
        text.set_visible(False)

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

    glob_start = time.perf_counter()
    pred, _ = ctcBeamSearch(global_softmaxes[i],
                            classes,
                            r_model,
                            30,
                            0.5,
                            0.9,
                            8)
    glob_end = time.perf_counter()
    print(pred)

    # Local decoding

    loc_start = time.perf_counter()
    local_preds = []
    for softmax in local_softmaxes[i]:
        pred, _ = ctcBeamSearch(softmax,
                                classes,
                                r_model,
                                30,
                                0.5,
                                0.9,
                                8)
        local_preds.append(pred)
    consensus = simple_assembly(local_preds)
    consensus_seq = index2base(np.argmax(consensus, axis=0))
    loc_end = time.perf_counter()
    print(consensus_seq)

    # Print results

    print(f"Global decoding took {glob_end - glob_start} seconds")
    print(f"Local decoding took {loc_end - loc_start} seconds")

if __name__ == "__main__":
    main()
