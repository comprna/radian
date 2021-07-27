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
    # d_config_file = sys.argv[3]
    # d_config = get_config(d_config_file)

    # Load RNA model

    # if d_config.use_rna_model == False:
    #     r_model = None
    # else:
    #     r_config_file = sys.argv[4]
    #     r_config = get_config(r_config_file)
    #     r_model_file = sys.argv[5]
    #     r_model = get_rna_prediction_model(r_model_file, r_config)

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
    
    # Decode without RNA model

    r_model= None
    classes = 'ACGT'
    for i, softmax in enumerate(global_softmaxes):
        pred, inds = ctcBeamSearch(softmax,
                                   classes,
                                   r_model,
                                   30,
                                   0.5,
                                   0.9,
                                   8)


        # Visualise softmax and decoding

        alignments = pairwise2.align.globalxx(gts[read_ids[i]], pred)
        for alignment in alignments:
            print(format_alignment(*alignment))

        fig, axs = plt.subplots(2, 1, sharex="all", figsize=(40,6))

        axs[0].imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
        axs[1].imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
        overlay_prediction(axs[1], pred, inds, 'blue')
        plt.show()

if __name__ == "__main__":
    main()