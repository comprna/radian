import json

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import matplotlib.pyplot as plt
import numpy as np

from beam_search_decoder import ctcBeamSearch
from utilities import setup_local

def overlay_prediction(plot, prediction, indices, color, offset=0):
    bases = ['A', 'C', 'G', 'T']
    for i, nt in enumerate(prediction):
        plot.text(indices[i], bases.index(nt)+offset, nt,  fontsize='x-large', color=color)

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

    # Load ground truth sequences

    gt_heart_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/heart/heart_read_labels_all.npy"
    with open(gt_heart_file, "rb") as f:
        gt_per_read = np.load(f, allow_pickle=True)

    # Load local softmaxes

    softmaxes_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/6_WriteSoftmaxes/softmaxes_per_read_heart.npy"
    with open(softmaxes_file, "rb") as f:
        softmaxes_per_read = np.load(f, allow_pickle=True)

    # Decode without RNA model

    r_model= None
    classes = 'ACGT'
    for i, read in enumerate(softmaxes_per_read):
        for j, softmax in enumerate(read):
            pred, inds = ctcBeamSearch(softmax,
                                       classes,
                                       r_model,
                                       30,
                                       0.5,
                                       0.9,
                                       8)

            # Visualise softmax and decoding

            alignments = pairwise2.align.globalxx(gt_per_read[i][j], pred)
            for alignment in alignments:
                print(format_alignment(*alignment))

            fig, axs = plt.subplots(2, 1, sharex="all", figsize=(40,6))

            axs[0].imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
            axs[1].imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
            overlay_prediction(axs[1], pred, inds, 'blue')
            plt.show()

if __name__ == "__main__":
    main()