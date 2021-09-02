import json
from statistics import mean

import numpy as np
import pandas as pd
from textdistance import levenshtein


def main():

    # Load guppy predictions

    preds_file = '/mnt/sda/rna-basecaller/experiments/decode/without-rna-model/benchmark/4_ParseFastqToTsv/heart_all_preds.tsv'
    preds_df = pd.read_csv(preds_file, sep='\t')
    preds_df = preds_df[preds_df.read != 'read']  # Remove erroneous header rows included in concatenation of fastq files

    # Replace all Us with Ts so sequences are comparable to GT

    preds_df['pred'] = preds_df['pred'].astype('string')
    preds_df.pred = preds_df.pred.str.replace('U', 'T')

    # Reverse the Guppy sequences

    preds_df.pred = preds_df.pred.str[::-1]

    # Load read IDs

    read_ids_file = "/mnt/sda/rna-basecaller/experiments/decode/global-vs-local/train-3-37/val_test/5_WriteToNumpy/heart/heart_read_ids.npy"
    with open(read_ids_file, "rb") as f:
        read_ids = np.load(f)

    # Load our predictions

    our_preds_file = 'decode-7-out.txt'
    our_preds = {}
    with open(our_preds_file, "r") as f:
        for line in f:
            cols = line.split('\t')
            read = cols[1]
            pred = cols[3]
            our_preds[read] = pred

    print(our_preds)

    # Load ground truth sequences

    gt_file = "/mnt/sda/rna-basecaller/experiments/decode/global-vs-local/train-3-37/val_test/4_Construct_GT_Label_Per_Read/heart_read_gt_labels.json"
    with open(gt_file, "r") as f:
        gts = json.load(f)

    n_better  = 0
    n_worse   = 0
    n_same    = 0
    our_eds   = []
    guppy_eds = []
    for i, read in enumerate(read_ids):
        gt = gts[read]
        our_pred = our_preds[i]
        our_ed = levenshtein.normalized_distance(gt, our_pred)
        our_eds.append(our_ed)

        guppy_pred = preds_df[preds_df['read']==read]['pred'].values[0]
        guppy_ed = levenshtein.normalized_distance(gt, guppy_pred)
        guppy_eds.append(guppy_ed)

        if our_ed < guppy_ed:
            n_better += 1
        elif our_ed > guppy_ed:
            n_worse += 1
        else:
            n_same += 1

    print(f"n_better: {n_better}")
    print(f"n_worse: {n_worse}")
    print(f"n_same: {n_same}")


    print(f"our ED mean: {mean(our_eds)}")
    print(f"guppy ED mean: {mean(guppy_eds)}")



if __name__ == "__main__":
    main()