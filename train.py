import argparse
import json
import sys
import yaml
from attrdict import AttrDict
from datagen import DataGenerator
from model import initialise_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

def compute_max_length(labels):
    max_len = 0
    for label in labels:
        if len(label) > max_len:
            max_len = len(label)
    return max_len

def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))

def get_partitions(config, dir):
    partitions = []
    for i in range(1, config.train.n_folds + 1):
        with open('{0}/partition-{1}.json'.format(dir, i), 'r') as f:
            partitions.append(json.load(f))
    return partitions

def get_labels(labels_file):
    with open(labels_file, 'r') as labels_f:
        return json.load(labels_f)

def train(checkpoint, epoch_to_resume, partition):
    partitions_dir = '/home/alex/OneDrive/phd-project/singleton-dataset-generation/dRNA/3_8_NNInputs/test'
    labels_file = '/home/alex/OneDrive/phd-project/singleton-dataset-generation/dRNA/3_8_NNInputs/test/labels.json'

    # Load params
    c = get_config('config.yaml')

    # Prepare data generators
    labels = get_labels(labels_file)
    max_label_length = compute_max_length(labels.values())
    partitions = get_partitions(c, partitions_dir)
    params = {'batch_size': c.train.batch_size,
              'window_size': c.data.window_size,
              'max_label_len': max_label_length,
              'shuffle': True}

    # Train using k-fold CV
    cv_scores = []
    for i, partition in enumerate(partitions):
        print("Training model {0}...".format(i))

        if checkpoint is not None:
            model = load_model(checkpoint)
            initial_epoch = epoch_to_resume
        else:
            model = initialise_model(c.model, c.train.opt, max_label_length)
            initial_epoch = 0

        val_score = train_on_partition(model, partition, initial_epoch,
            labels, params, c)
        print("Model {0} scores: Loss = {1}".format(i, val_score))
        cv_scores.append(val_score)
    print(cv_scores)

def train_on_partition(model, partition, init_epoch, labels, gen_params, 
    config):
    c = config

    train_generator = DataGenerator(partition['train'], labels, **gen_params)
    val_generator = DataGenerator(partition['val'], labels, **gen_params)

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
        callbacks = callbacks_list)
        # use_multiprocessing = True,
        # workers = 2)
    
    return model.evaluate(
        x = val_generator, 
        steps = n_val_signals // c.train.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="path to weights checkpoint to load")
    parser.add_argument("-e", "--initial_epoch", help="epoch to resume training at")
    parser.add_argument("-p", "--partition", help="partition to resume training with")
    args = parser.parse_args()

    if args.checkpoint is not None:
        assert args.initial_epoch is not None
        assert args.partition is not None

    # train(args.checkpoint, args.initial_epoch, args.partition)
    train(None, None, None)
