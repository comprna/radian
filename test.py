import matplotlib.pyplot as plt
import numpy as np
from tensorflow.io.gfile import glob
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from data import get_dataset
from edit_distance import compute_mean_edit_distance
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()
    config = get_config('/home/150/as2781/rnabasecaller/config.yaml')

    # Get test data
    test_files = glob("/g/data/xc17/Eyras/alex/rna-basecaller/shards/debugging/CATTTTATCTCTGGGTCATT_GCCTACTTCGTCTATCACTCCT/*.tfrecords")
    test_dataset = get_dataset(test_files, config, val=True)

    # Load finalized model
    saved_filepath = '/g/data/xc17/Eyras/alex/rna-basecaller/train-test/model-53.h5'
    model = get_prediction_model(saved_filepath, config)

    # TODO: Assemble into reads

    compute_mean_edit_distance(model, test_dataset, verbose=True)

if __name__ == "__main__":
    main()

    # Refs:
    # https://www.programcreek.com/python/example/122027/keras.backend.ctc_decode
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
    # https://machinelearningmastery.com/train-final-machine-learning-model/
    # https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/
