import cProfile
import json
import os
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from textdistance import levenshtein

from beam_search_decoder import ctcBeamSearch
from rna_model import get_rna_prediction_model
from utilities import get_config, setup_local

def main():
    # Load tsv file containing predictions and ground truth
    with open(sys.argv[1], "r") as f_in:
        df = pd.read_csv(f_in, sep='\t', header=None, names=["read", "gt", "pred"])
        df['ed'] = df.apply(lambda row: levenshtein.normalized_distance(row['gt'], row['pred']), axis=1)
        with open(f"{sys.argv[1]}-ED.tsv", "w") as f_out:
            df.to_csv(f_out, sep='\t')

if __name__ == "__main__":
    main()