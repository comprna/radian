import glob
import os
import time

import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.io import gfile
from tensorflow.keras import backend as K
from textdistance import levenshtein

from data import get_dataset
from evaluate import run_distributed_predict_greedy, predict_greedy_serial
from model import get_prediction_model
from utilities import get_config, setup_local

def run_mirrored_strategy(model_file, config, data_files):
    # Create a strategy to use all available GPUs.
    strategy = MirroredStrategy()

    # Create the model inside the strategy's scope so that it is a
    # mirrored variable.
    with strategy.scope():
        model = get_prediction_model(model_file, config)

    # Create a distributed dataset based on the strategy.
    dataset = get_dataset(data_files, config.train.batch_size, val=True)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # Run the distributed prediction.
    return run_distributed_predict_greedy(strategy, dist_dataset, model)

def run_serial(model_file, config, data_files):
    model = get_prediction_model(model_file, config)
    dataset = get_dataset(data_files, config.train.batch_size, val=True)

    return predict_greedy_serial(model, dataset, verbose=True)

def main():
    # MacBook Pro

    # setup_local()
    # config = get_config('/Users/alexsneddon/Documents/phd/rnabasecaller/config.yaml')
    # data_files = gfile.glob("/Users/alexsneddon/Documents/phd/rnabasecaller/debug/*.tfrecords")
    # model_file = "/Users/alexsneddon/Documents/phd/rnabasecaller/model-01.h5"

    # Local Desktop

    # setup_local()
    # config = get_config('/home/alex/Documents/rnabasecaller/config.yaml')
    # data_files = gfile.glob("/mnt/sda/singleton-dataset-generation/dRNA/4_8_NNInputs/0_2_CreateTFRecords/2_WriteTFRecords/shards/local_testing/val/*.tfrecords")
    # model_file = "/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/train-1/model-01.h5"

    # Gadi

    config = get_config('/home/150/as2781/rnabasecaller/config.yaml')
    data_files = gfile.glob("/g/data/xc17/Eyras/alex/working/test_shards/val/*.tfrecords")
    model_file = "/g/data/xc17/Eyras/alex/working/rna-basecaller/4_8_NNInputs/train-1/model-01.h5"


    # BENCHMARKING
    # NB: Using time.time() only gives an approximate time, but since
    # the running times are long this is good enough here. More accurate
    # benchmarking can be done with the timeit module.


    # predictions = run_mirrored_strategy(model_file, config, data_files)
    predictions = run_serial(model_file, config, data_files)

    # print(predictions)

    # eds = []
    # for p in predictions:
    #     ed = levenshtein.normalized_distance(p[0], p[1])
    #     eds.append(ed)


    
    # print("Starting benchmarking...")
    # start = time.time()
    # predictions = []
    # for chunk in dist_dataset:
    #     predictions.append(strategy.run(predict_greedy_opt, args=(model, chunk)))

    # prediction_end = time.time()
    # mean_ed = compute_mean_ed(predictions)
    # mean_end = time.time()

    # prediction_time = prediction_end - start
    # mean_time = mean_end - prediction_end
    # total_time = prediction_time + mean_time

    # print("Time to predict: {}".format(prediction_time))
    # print("Time to compute mean ED: {}".format(mean_time))
    # print("Total time: {}".format(total_time))
    # print("Mean ED: {}".format(mean_ed))

if __name__ == "__main__":
    main()