import glob
import os
import time

from tensorflow.io import gfile

from data import get_dataset
from evaluate import predict_greedy, compute_mean_ed
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    # setup_local()
    # config = get_config('/home/alex/Documents/rnabasecaller/config.yaml')
    config = get_config('/home/150/as2781/rnabasecaller/config.yaml')

    # data_files = gfile.glob("/mnt/sda/singleton-dataset-generation/dRNA/4_8_NNInputs/0_2_CreateTFRecords/2_WriteTFRecords/shards/val/*.tfrecords")
    data_files = gfile.glob("/g/data/xc17/Eyras/alex/working/test_shards/val/*.tfrecords")
    dataset = get_dataset(data_files, config.train.batch_size, val=True)

    # model_file = "/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/train-1/model-01.h5"
    model_file = "/g/data/xc17/Eyras/alex/working/rna-basecaller/4_8_NNInputs/train-1/model-01.h5"
    model = get_prediction_model(model_file, config)

    # NB: Using time.time() only gives an approximate time, but since
    # the running times are long this is good enough here. More accurate
    # benchmarking can be done with the timeit module.

    print("Starting benchmarking...")

    start = time.time()
    predictions = predict_greedy(model, dataset)
    prediction_end = time.time()
    mean_ed = compute_mean_ed(predictions)
    mean_end = time.time()

    prediction_time = prediction_end - start
    mean_time = mean_end - prediction_end
    total_time = prediction_time + mean_time

    print("Time to predict: {}".format(prediction_time))
    print("Time to compute mean ED: {}".format(mean_time))
    print("Total time: {}".format(total_time))
    print("Mean ED: {}".format(mean_ed))

if __name__ == "__main__":
    main()