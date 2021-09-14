import json
import sys

import numpy as np
import pandas as pd
from textdistance import levenshtein


def main():

    # Load guppy predictions

    preds_file = sys.argv[1]
    preds_df = pd.read_csv(preds_file, sep='\t')
    preds_df = preds_df[preds_df.read != 'read']  # Remove erroneous header rows included in concatenation of fastq files

    # Replace all Us with Ts so sequences are comparable to GT

    preds_df['pred'] = preds_df['pred'].astype('string')
    preds_df.pred = preds_df.pred.str.replace('U', 'T')

    # Reverse the Guppy sequences

    preds_df.pred = preds_df.pred.str[::-1]

    # Load read IDs

    read_ids_file = sys.argv[2]
    with open(read_ids_file, "rb") as f:
        read_ids = np.load(f)

    # Load ground truth sequences

    gt_file = sys.argv[3]
    with open(gt_file, "r") as f:
        gts = json.load(f)

    print("read\tgt\tpred\ted")
    for read in read_ids:
        gt = gts[read]
        pred = preds_df[preds_df['read']==read]['pred'].values[0]
        ed = levenshtein.normalized_distance(gt, pred)
        print(f"{read}\t{gt}\t{pred}\t{ed}")



if __name__ == "__main__":
    main()