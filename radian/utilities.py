import ast
import json
import sys
import time

import yaml
from attrdict import AttrDict
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.io.gfile import glob
from tensorflow.keras import backend as K

from data import get_dataset


def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))

def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    # for epoch_num in range(num_epochs):
    for s in dataset:
        pass
    tf.print("execution time: {0}".format(time.perf_counter() - start_time))

def count_n_steps_per_epoch(dataset):
    n = 0
    for sample in dataset:
        n += 1
        print(n)
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
    # tf.config.experimental_run_functions_eagerly(True)

def print_dataset(shards_dir):
    data_files = glob("{0}/*.tfrecords".format(shards_dir))

    with open('config.yaml') as config_file:
        config = AttrDict(yaml.load(config_file, Loader=yaml.Loader))

    dataset = get_dataset(data_files, config.train.batch_size, val=False)

    for batch in dataset:
        # print(batch)
        inputs = batch[0]
        signal_batch = inputs['inputs']
        label_batch = inputs['labels']
        signal = signal_batch[0]
        print(signal_batch)
        print(signal)

        fig, axs = plt.subplots(10, 2, sharey='all')
        for i in range(20):
            print(label_batch[i])
            # print("i: {0}, i/10: {1}, i%10: {2}".format(i, int(i/10), i%10))
            axs[i%10, int(i/10)].plot(signal_batch[i])

        # plt.plot(signal)
        plt.show()

def label_to_sequence(label, label_length):
    label = label[:label_length]
    bases = ['A', 'C', 'G', 'T']
    label = list(map(lambda b: bases[b], label))
    return "".join(label)

def to_int_list(float_tensor):
    return K.cast(float_tensor, "int32").numpy()

def get_label_stats(dataset):
    label_count = {}
    
    for batch in dataset:
        inputs = batch[0]
        label_batch = inputs['labels']
        label_lengths = batch[0]["label_length"]

        for i, label in enumerate(label_batch):
            label = to_int_list(label)
            label = label_to_sequence(label, label_lengths[i])

            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

    with open('labels.json', 'w') as f:
        json.dump(label_count, f)


def print_same_label_signals(dataset):

    # target = "AAAAAAA"
    # target = "AAAAA"
    target = "AGACTCCGAACATCCTCCCATTT"
    target_signals = []

    for batch in dataset:
        inputs = batch[0]
        signal_batch = inputs['inputs']
        label_batch = inputs['labels']
        label_lengths = batch[0]["label_length"]

        for i, label in enumerate(label_batch):
            label = to_int_list(label)
            label = label_to_sequence(label, label_lengths[i])

            if label == target:
                target_signals.append(signal_batch[i])
                print(len(target_signals))
        
        if len(target_signals) == 6:
            break

    fig, axs = plt.subplots(3, 2, sharey='all')
    for i in range(len(target_signals)):
        axs[i%3, int(i/3)].plot(target_signals[i])

    plt.suptitle("Signals for {}".format(target))
    plt.show()

if __name__ == "__main__":
    data_files = glob(f"{sys.argv[1]}/*.tfrecords")

    with open(sys.argv[2]) as config_file:
        config = AttrDict(yaml.load(config_file, Loader=yaml.Loader))

    dataset = get_dataset(data_files, config.train.batch_size, val=True)

    count_n_steps_per_epoch(dataset)