import json
from statistics import mean

import numpy as np
from textdistance import levenshtein

def main():
    # Load read IDs

    read_ids_hek293_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/hek293/hek293_read_ids.npy"
    with open(read_ids_hek293_file, "rb") as f:
        read_ids_hek293 = np.load(f)

    # Load ground truth sequences

    gt_hek293_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/4_Construct_GT_Label_Per_Read/hek293_read_gt_labels.json"
    with open(gt_hek293_file, "r") as f:
        gt_hek293 = json.load(f)

    # Load local predictions

    preds_hek293_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/7_TestLocalDecoding/2_AssembleLocalPreds/read_assembled_preds_hek293.npy"
    with open(preds_hek293_file, "rb") as f:
        preds_hek293 = np.load(f)

    # Compute average edit distance

    eds_hek293 = []
    for i, read in enumerate(read_ids_hek293):
        gt = gt_hek293[read]
        pred = preds_hek293[i]
        ed = levenshtein.normalized_distance(gt, pred)
        eds_hek293.append(ed)
        print(f"{read}\t{gt}\t{pred}\t{ed}")
    print(f"Average ED: {mean(eds_hek293)}")

if __name__ == "__main__":
    main()