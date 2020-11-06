import argparse
import h5py
import json
import numpy as np
import sys
import tensorflow as tf
import time
import yaml
from attrdict import AttrDict
from datagen import DataGenerator
from model import initialise_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# Profiling
from datetime import datetime
from packaging import version
import os

# Computed elsewhere
MAX_LABEL_LEN = 46
STEPS_PER_EPOCH = 41407
WINDOWS_PER_SHARD = 50000

def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))

def read_tfrecord(example_batch):
    features = {
        'signal': tf.io.FixedLenFeature([512], tf.float32), # TODO: Remove hardcoding
        'label': tf.io.VarLenFeature(tf.float32),
        'signal_length': tf.io.FixedLenFeature([], tf.int64), # shape [] means scalar
        'label_length': tf.io.FixedLenFeature([], tf.int64)
    }

    tf_record = tf.io.parse_example(example_batch, features)
    
    signal = tf_record['signal']
    label = tf.sparse.to_dense(tf_record['label']) # VarLenFeature decoding required
    signal_length = tf_record['signal_length']
    label_length = tf_record['label_length']

    inputs = {
        'inputs': signal,
        'labels': label,
        'input_length': signal_length,
        'label_length': label_length
    }
    outputs = {'ctc': np.zeros([512])} # TODO: Remove hardcoding

    return inputs, outputs

def get_batched_dataset(shard_files, config, val = False):
    # AUTO = tf.data.experimental.AUTOTUNE
    AUTO = 48

    # Create a dataset that is made of shuffled shards
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    shards = tf.data.Dataset.from_tensor_slices(shard_files)
    shards = shards.with_options(option_no_order)
    shards = shards.shuffle(buffer_size = tf.cast(tf.shape(shard_files)[0], tf.int64))
    if val == False:
        shards = shards.repeat()

    # Create a TFRecordsDataset that reads shard files in random order
    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=48, 
                                num_parallel_calls=AUTO)
    dataset = dataset.shuffle(buffer_size=WINDOWS_PER_SHARD+1)
    dataset = dataset.batch(config.train.batch_size)
    dataset = dataset.map(map_func = read_tfrecord,
                          num_parallel_calls = AUTO)
    dataset = dataset.cache() # Cache after map
    dataset = dataset.prefetch(AUTO)
    return dataset

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

def train(shards_dir, checkpoint, epoch_to_resume, config_file):
    c = get_config(config_file)

    # # Benchmarking
    # train_filenames = tf.io.gfile.glob("{0}/*.tfrecords".format(shards_dir))
    # train_dataset = get_batched_dataset(train_filenames, c, val=True)
    # benchmark(train_dataset)

    train_filenames = tf.io.gfile.glob("{0}/train/*.tfrecords".format(shards_dir))
    train_dataset = get_batched_dataset(train_filenames, c, val=False)

    val_filenames = tf.io.gfile.glob("{0}/val/*.tfrecords".format(shards_dir))
    val_dataset = get_batched_dataset(val_filenames, c, val=True)

    strategy = tf.distribute.MirroredStrategy()
    print("\n\n\n\n\nNumber of devices: {}\n\n\n\n\n".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        if checkpoint is not None:
            model = load_model(checkpoint)
            initial_epoch = epoch_to_resume
            print("Loaded checkpoint {0}".format(checkpoint))
        else:
            model = initialise_model(c.model, c.train.opt, MAX_LABEL_LEN)
            initial_epoch = 0

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

    checkpoint_path = "model-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, 
                                 monitor="val_loss",
                                 verbose=1,
                                 mode="min")
    callbacks_list = [checkpoint, tboard_callback]

    model.summary()
    model.fit(train_dataset,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = c.train.n_epochs,
        initial_epoch = initial_epoch,
        validation_data = val_dataset,
        validation_freq = c.train.val_freq,
        verbose = 1,   # Dev
        # verbose = 2, # Testing
        callbacks = callbacks_list)
    
    score = model.evaluate(x = val_dataset)
    print(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="path to weights checkpoint to load")
    parser.add_argument("-e", "--initial_epoch", help="epoch to resume training at")
    parser.add_argument("-g", "--config-file", help="file containing model config params")
    parser.add_argument("-s", "--shards-dir", help="directory containing train/val/test shard files")
    args = parser.parse_args()

    if args.checkpoint is not None:
        assert args.initial_epoch is not None

    # Running locally:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # tf.compat.v1.disable_eager_execution()

    train(args.shards_dir,
          args.checkpoint, 
          args.initial_epoch, 
          args.config_file)
    # train('/mnt/sda/singleton-dataset-generation/dRNA/3_8_NNInputs/tfrecord_approach/shards', None, None, 'config.yaml')