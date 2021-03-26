import cProfile
import glob
import os
import time

import tensorflow as tf
from tensorflow.io import gfile
from tensorflow.keras import backend as K
from textdistance import levenshtein

from data import get_dataset
from evaluate import predict_greedy, compute_mean_ed
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    # MacBook Pro

    setup_local()
    # config = get_config('/Users/alexsneddon/Documents/phd/rnabasecaller/config.yaml')
    # data_files = gfile.glob("/Users/alexsneddon/Documents/phd/rnabasecaller/debug/*.tfrecords")
    # model_file = "/Users/alexsneddon/Documents/phd/rnabasecaller/model-01.h5"

    # Local Desktop

    # setup_local()
    # config = get_config('/home/alex/Documents/rnabasecaller/config.yaml')
    # data_files = gfile.glob("/mnt/sda/singleton-dataset-generation/dRNA/4_8_NNInputs/0_2_CreateTFRecords/2_WriteTFRecords/shards/local_testing/val/*.tfrecords")
    # model_file = "/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/train-1/model-01.h5"

    # Gadi

    # tf.config.experimental_run_functions_eagerly(True)
    # config = get_config('/home/150/as2781/rnabasecaller/config.yaml')
    # data_files = gfile.glob("/g/data/xc17/Eyras/alex/working/test_shards/val/*.tfrecords")
    # model_file = "/g/data/xc17/Eyras/alex/working/rna-basecaller/4_8_NNInputs/train-1/model-01.h5"

    # MirroredStrategy significantly increased edit distance calc time

    # Edit distance https://github.com/cyprienruffino/CTCModel/blob/992e771937c94843a345dadc50770866b290e167/keras_ctcmodel/CTCModel.py

    # dataset = get_dataset(data_files, config.train.batch_size, val=True)

    # def callback():
    #     model = get_prediction_model(model_file, config) # Actually get_evaluation_model
    #     predictions = predict_greedy(model, dataset, verbose=True)
    #     mean_ed = compute_mean_ed(predictions)
    
    cProfile.run('callback()', sort='cumtime')

def callback():
    config = get_config('/Users/alexsneddon/Documents/phd/rnabasecaller/config.yaml')
    data_files = gfile.glob("/Users/alexsneddon/Documents/phd/rnabasecaller/debug/*.tfrecords")
    model_file = "/Users/alexsneddon/Documents/phd/rnabasecaller/model-01.h5"
    dataset = get_dataset(data_files, config.train.batch_size, val=True)

    model = get_prediction_model(model_file, config) # Actually get_evaluation_model
    predictions = predict_greedy(model, dataset, verbose=False)
    mean_ed = compute_mean_ed(predictions)


if __name__ == "__main__":
    # cProfile.run('callback()')
    main()