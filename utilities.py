import ast
import csv
import time
import yaml
from attrdict import AttrDict
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.io.gfile import glob

from data import get_dataset

def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    # for epoch_num in range(num_epochs):
    for s in dataset:
        pass
    tf.print("execution time: {0}".format(time.perf_counter() - start_time))

def count_training_size(dataset):
    n = 0
    for sample in dataset:
        n += 1
    print(n)

def get_data_info():
    with open('hek293-fold1.csv', "r") as f:
          i = 0
          max_len = 0
          while True:
              line = f.readline()
              i += 1
              # Move back to beginning of file if at end
              if line == "":
                  print("End of file at line {0}".format(i))
                  break

              signal, sequence = line.split('\t')
              sequence = ast.literal_eval(sequence)
              
              len_sequence = len(sequence)
              if len_sequence > max_len:
                  max_len = len_sequence
          
          print("Maximum sequence length: {0}".format(max_len))

def setup_local():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

def print_dataset():
    shards_dir = '/mnt/sda/singleton-dataset-generation/dRNA/3_8_NNInputs/tfrecord_approach/shards'
    train_files = glob("{0}/train/*.tfrecords".format(shards_dir))

    with open('config.yaml') as config_file:
        config = AttrDict(yaml.load(config_file, Loader=yaml.Loader))

    dataset = get_dataset(train_files, config, val=False)

    for batch in dataset:
        # print(batch)
        inputs = batch[0]
        signal_batch = inputs['inputs']
        signal = signal_batch[0]
        print(signal_batch)
        print(signal)

        fig, axs = plt.subplots(10, 2, sharey='all')
        for i in range(20):
            # print("i: {0}, i/10: {1}, i%10: {2}".format(i, int(i/10), i%10))
            axs[i%10, int(i/10)].plot(signal_batch[i])

        # plt.plot(signal)
        plt.show()

if __name__ == "__main__":
    print_dataset()