import json
import sys
import yaml
from attrdict import AttrDict
from datagen import DataGenerator
from model import build_model
from tensorflow import Variable
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers import Adam

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

def get_optimizer(config):
    step = Variable(0, trainable = False)
    boundaries = [int(config.max_steps * bound) for bound in config.boundaries]
    values = [config.init_rate * decay for decay in config.decays]
    learning_rate_fn = PiecewiseConstantDecay(boundaries, values)
    return Adam(learning_rate=learning_rate_fn(step))

def train():
    partitions_dir = '/home/alex/OneDrive/phd-project/singleton-dataset-generation/dRNA/3_8_NNInputs'
    labels_file = '/home/alex/OneDrive/phd-project/singleton-dataset-generation/dRNA/3_8_NNInputs/labels.json'

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

        train_generator = DataGenerator(partition['train'], labels, **params)
        val_generator = DataGenerator(partition['val'], labels, **params)

        n_train_signals = len(partition['train'])
        n_val_signals = len(partition['val'])

        model = build_model(c.model, max_label_length)
        optimizer = get_optimizer(c.train.opt)
        model.compile(optimizer = optimizer,
                      loss = {'ctc': lambda labels, y_pred: y_pred})

        model.summary()
        model.fit(
            x = train_generator,
            validation_data=val_generator,
            validation_steps = n_val_signals // c.train.batch_size,
            validation_freq = c.train.val_freq,
            steps_per_epoch = n_train_signals // c.train.batch_size,
            epochs = c.train.n_epochs)
            # use_multiprocessing = True,
            # workers = 2)
        
        score = model.evaluate(
            x = val_generator,
            steps = n_val_signals // c.train.batch_size)
        print("Model {0} scores: Loss = {1}".format(i, score))
        cv_scores.append(score)
    print(cv_scores)

    # TODO(save model)

if __name__ == "__main__":
    train()
