import argparse
import h5py
import json
import sys
import tensorflow as tf
import yaml
from attrdict import AttrDict
from datagen import DataGenerator
from model import initialise_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

MAX_LABEL_LEN = 46

def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))

def get_partitions(config, dir):
    partitions = []
    for i in range(1, config.train.n_folds + 1):
        with open('{0}/partition-{1}.json'.format(dir, i), 'r') as f:
            partitions.append(json.load(f))
    return partitions

def train_on_partition(model, partition, init_epoch, data_file, gen_params, 
    config):
    c = config

    train_generator = DataGenerator(partition['train'], data_file, **gen_params)
    val_generator = DataGenerator(partition['val'], data_file, **gen_params)

    n_train_signals = len(partition['train'])
    n_val_signals = len(partition['val'])

    checkpoint_path = "model-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, 
                                 monitor="val_loss",
                                 verbose=1,
                                 mode="min")
    callbacks_list = [checkpoint]

    model.summary()
    model.fit(
        x = train_generator,
        validation_data=val_generator,
        validation_steps = n_val_signals // c.train.batch_size,
        validation_freq = c.train.val_freq,
        steps_per_epoch = n_train_signals // c.train.batch_size,
        epochs = c.train.n_epochs,
        initial_epoch = init_epoch,
        # verbose = 1,   # Dev
        verbose = 2, # Testing
        callbacks = callbacks_list)

    return model.evaluate(
        x = val_generator, 
        steps = n_val_signals // c.train.batch_size)

def train(checkpoint, epoch_to_resume, partition_to_resume, 
    partitions_dir, data_file, config_file):
    # Load params
    c = get_config(config_file)

    # Prepare data generators
    partitions = get_partitions(c, partitions_dir)
    params = {'batch_size': c.train.batch_size,
              'window_size': c.data.window_size,
              'max_label_len': MAX_LABEL_LEN,
              'shuffle': True}

    # Train using k-fold CV
    cv_scores = []
    for i, partition in enumerate(partitions):
        if partition_to_resume is not None and i < partition_to_resume - 1:
            continue

        print("Training model with partition {0}...".format(i))

        if checkpoint is not None:
            model = load_model(checkpoint)
            initial_epoch = epoch_to_resume
            print("Loaded checkpoint {0}".format(checkpoint))
        else:
            model = initialise_model(c.model, c.train.opt, MAX_LABEL_LEN)
            initial_epoch = 0

        val_score = train_on_partition(model, partition, initial_epoch,
            data_file, params, c)
        print("Model {0} scores: Loss = {1}".format(i, val_score))
        cv_scores.append(val_score)
    print(cv_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="path to weights checkpoint to load")
    parser.add_argument("-e", "--initial_epoch", help="epoch to resume training at")
    parser.add_argument("-p", "--partition", help="partition to resume training with")
    parser.add_argument("-d", "--partitions-dir", help="directory containing partitions")
    parser.add_argument("-f", "--data-file", help="h5 file containing training data")
    parser.add_argument("-g", "--config-file", help="file containing model config params")
    args = parser.parse_args()

    if args.checkpoint is not None:
        assert args.initial_epoch is not None
        assert args.partition is not None

    config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1 , 'CPU': 8}) 
    sess = tf.compat.v1.Session(config=config) 
    tf.compat.v1.keras.backend.set_session(sess)

    from tensorflow.python.client import device_lib
    print("Local devices: {0}".format(device_lib.list_local_devices()))

    train(args.checkpoint, 
          args.initial_epoch, 
          args.partition, 
          args.partitions_dir, 
          args.data_file, 
          args.config_file)