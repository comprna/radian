import json

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import matplotlib.pyplot as plt
import numpy as np

from beam_search_decoder import ctcBeamSearch
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

    # Decode without RNA model

    r_model= None
    classes = 'ACGT'
    for i, read in enumerate(local_softmaxes):
        
        # # Global decoding

        # pred, inds = ctcBeamSearch(global_softmaxes[i],
        #                            classes,
        #                            r_model,
        #                            30,
        #                            0.5,
        #                            0.9,
        #                            8)

        # # Visualise alignment

        # alignments = pairwise2.align.globalxx(global_gts[read_ids[i]], pred)
        # for alignment in alignments:
        #     print(format_alignment(*alignment))
        #     break # Only print the first alignment for now

        # Local decoding

        for j, softmax in enumerate(read):
            pred, inds = ctcBeamSearch(softmax,
                                       classes,
                                       r_model,
                                       30,
                                       0.5,
                                       0.9,
                                       8)

            # Visualise alignment

            alignments = pairwise2.align.globalxx(local_gts[i][j], pred)
            for alignment in alignments:
                print(format_alignment(*alignment))
                break # Only print the first alignment for now

            # Visualise softmax and decoding

            plt.figure(figsize=(20, 3))
            plt.imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")

            # Save plot in chunks

            chunk_size = 341
            n_chunks = int(len(softmax) / chunk_size)
            for k in range(n_chunks):
                start = k * chunk_size
                end = start + chunk_size
                plt.xlim(start, end)
                texts = overlay_prediction(plt, pred, inds, start, end, 'blue')
                plt.savefig(f"read-{read_ids[i]}-{j}-{k}.png", bbox_inches="tight", pad_inches=0)
                clear_texts(texts)
            plt.show()


if __name__ == "__main__":
    main()
