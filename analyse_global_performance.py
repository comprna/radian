import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from textdistance import levenshtein

from beam_search_decoder import ctcBeamSearch

STEP_SIZE = 128
WINDOW_LEN = 1024
OVERLAP = WINDOW_LEN-STEP_SIZE
N_BASES = 4
N_WINDOWS = 30

def overlay_prediction(plot, prediction, indices, color="blue", offset=0):
    bases = ['A', 'C', 'G', 'T']
    for i, nt in enumerate(prediction[:len(indices)]):
        plot.text(indices[i], bases.index(nt)+offset, nt,  fontsize='x-large', color=color)

def visualise_assembly(softmax_windows, global_softmax, pred_global, pred_global_i, pred_loc, gt, ed_glob, ed_loc):
    softmax_windows = softmax_windows[:N_WINDOWS]
    len_global = WINDOW_LEN + (N_WINDOWS - 1) * STEP_SIZE
    global_softmax = global_softmax[:len_global]
    pad_size_before = 0
    pad_size_after = len_global - WINDOW_LEN
    padded_windows = []
    for softmax in softmax_windows:
        padded_window = np.pad(softmax, ((pad_size_before, pad_size_after), (0,0)), "constant")
        padded_windows.append(padded_window)
        pad_size_before += STEP_SIZE
        pad_size_after -= STEP_SIZE
    plot_assembly(padded_windows, global_softmax, pred_global, pred_global_i, pred_loc, gt, ed_glob, ed_loc)

def plot_assembly(padded_windows, global_softmax, pred_global, pred_global_i, pred_loc, gt, ed_glob, ed_loc):
    # Only show first n windows for ease of viewing
    n_windows_to_view = 5
    _, axs = plt.subplots(n_windows_to_view+1, 1, sharex="all")
    for i, softmax in enumerate(padded_windows):
        if i >= n_windows_to_view:
            break
        axs[i].imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
    axs[-1].imshow(np.transpose(global_softmax), cmap="gray_r", aspect="auto")
    overlay_prediction(axs[-1], pred_global, pred_global_i)
    axs[-1].text(50, 6, f"GT:       {gt}", fontsize="x-large", color="orange", family="monospace")
    axs[-1].text(50, 7, f"Gl: {ed_glob:.3f} {pred_global}", fontsize="x-large", color="blue", family="monospace")
    axs[-1].text(50, 8, f"Lo: {ed_loc:.3f} {pred_loc}", fontsize="x-large", color="green", family="monospace")

    plt.show()

def main():
    # Load read IDs

    read_ids_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/hek293/hek293_read_ids.npy"
    with open(read_ids_file, "rb") as f:
        read_ids = np.load(f)
    
    # Load ground truth sequences

    gt_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/4_Construct_GT_Label_Per_Read/hek293_read_gt_labels.json"
    with open(gt_file, "r") as f:
        gts = json.load(f)

    # Load local softmaxes

    data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test"
    with open(f"{data_dir}/6_WriteSoftmaxes/softmaxes_per_read_hek293.npy", "rb") as f:
        local_softmaxes = np.load(f, allow_pickle=True)

    # Load global softmaxes

    data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/8_TestGlobalDecoding/1_CombineSoftmaxes/hek293"
    npy_paths = sorted(Path(data_dir).iterdir(), key=os.path.getmtime)
    with open(npy_paths[0], "rb") as f:
        global_softmaxes = np.load(f, allow_pickle=True)
    for path in npy_paths[1:]:
        with open(path, "rb") as f:
            global_softmaxes = np.concatenate((global_softmaxes, np.load(f, allow_pickle=True)))

    # Load local predictions

    preds_loc_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/7_TestLocalDecoding/2_AssembleLocalPreds/read_assembled_preds_hek293.npy"
    with open(preds_loc_file, "rb") as f:
        preds_loc = np.load(f)
    
    # Load global predictions

    preds_glob_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/8_TestGlobalDecoding/2_DecodeGlobalSoftmax/global_preds_hek293.npy"
    with open(preds_glob_file, "rb") as f:
        preds_glob = np.load(f)

    # For each read
    for i, read in enumerate(read_ids):
        # Global decode
        global_softmax = global_softmaxes[i][:1600]
        _, pred_glob_i = ctcBeamSearch(global_softmax, 'ACGT', None, None)

        # Info needed for visualisation
        gt = gts[read]
        pred_loc = preds_loc[i]
        pred_glob = preds_glob[i]
        ed_glob = levenshtein.normalized_distance(gt, pred_glob)
        ed_loc = levenshtein.normalized_distance(gt, pred_loc)
        local_softmax = local_softmaxes[i]

        # Visualise global decoding
        visualise_assembly(local_softmax, global_softmax, pred_glob, pred_glob_i, pred_loc, gt, ed_glob, ed_loc)


if __name__ == "__main__":
    main()