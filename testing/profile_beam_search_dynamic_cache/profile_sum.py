import cProfile
import json

import numpy as np
from textdistance import levenshtein

from decode import beam_search
from testing.obsolete.rna_model_lstm import get_rna_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()
    # callback()

    cProfile.run("callback()", sort="cumtime")

def callback():
    a = np.array([0.1,0.2,0.3,0.4])
    for i in range(1000000):
        np.sum(a)

if __name__ == "__main__":
    main()