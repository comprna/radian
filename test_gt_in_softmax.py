import json

from matplotlib import pyplot as plt
import numpy as np
from textdistance import levenshtein

from beam_search_decoder import beam_search
from utilities import setup_local

def overlay_prediction(plot, prediction, indices, color, offset=0):
    bases = ['A', 'C', 'G', 'T']
    for i, nt in enumerate(prediction):
        plot.text(indices[i], bases.index(nt)+offset, nt,  fontsize='x-large', color=color)

def main():
    # Run locally or on gadi

    setup_local()
    base_dir = local_dir = '/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/copied_files'
    # gadi_dir = '/g/data/xc17/Eyras/alex/working/with-rna-model/global/all_val/copied_files'
    # base_dir = sys.argv[1]

    dataset = "heart"

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
    for i, read in enumerate(local_softmaxes):
        for j, softmax in enumerate(read):
            
            # Convert GT to tuple
            gt = ()
            for b in local_gts[i][j]:
                gt += (classes.index(b),)

            pred, inds = beam_search(softmax,
                                classes,
                                100,
                                None,
                                None,
                                None,
                                None,
                                gt)
            fig, axs = plt.subplots(2, 1, sharex="all", figsize=(40,6))

            axs[0].imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
            axs[1].imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
            overlay_prediction(axs[1], pred, inds, 'blue')
            plt.show()


if __name__ == "__main__":
    main()