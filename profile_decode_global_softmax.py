import cProfile

import numpy as np

from beam_search_decoder import ctcBeamSearch
from rna_model import get_rna_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()

    cProfile.run("callback()", sort="cumtime")

def callback():

    # Set up RNA model
    r_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/r-config-37.yaml'
    r_model_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/r-train-37-model-03.h5'
    r_config = get_config(r_config_file)
    r_model = get_rna_prediction_model(r_model_file, r_config)
    factor = 0.5
    npy_file = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/8_TestGlobalDecoding/1_CombineSoftmaxes/hek293/hek293_read_global_softmaxes_100.npy"

    with open(npy_file, "rb") as f:
        read_global_softmaxes = np.load(f, allow_pickle=True)

    classes = 'ACGT'
    ctcBeamSearch(read_global_softmaxes[0], classes, r_model, None)

if __name__ == "__main__":
    main()