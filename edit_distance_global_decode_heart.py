import json
from statistics import mean

import numpy as np
from textdistance import levenshtein

def main():
    # Load read IDs

    read_ids_heart_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/5_WriteToNumpy/heart/heart_read_ids.npy"
    with open(read_ids_heart_file, "rb") as f:
        read_ids_heart = np.load(f)

    # Load ground truth sequences

    gt_heart_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/4_Construct_GT_Label_Per_Read/heart_read_gt_labels.json"
    with open(gt_heart_file, "r") as f:
        gt_heart = json.load(f)

    # Load global predictions

    preds_heart_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/8_TestGlobalDecoding/2_DecodeGlobalSoftmax/global_preds_heart.npy"
    with open(preds_heart_file, "rb") as f:
        preds_heart = np.load(f)

    # Compute average edit distance

    eds_heart = []
    for i, read in enumerate(read_ids_heart):
        gt = gt_heart[read]
        pred = preds_heart[i]
        ed = levenshtein.normalized_distance(gt, pred)
        eds_heart.append(ed)
        print(f"{read}\t{ed}")
    print(f"Average ED: {mean(eds_heart)}")

if __name__ == "__main__":
    main()