import json
import sys
import tensorflow as tf
import tcn
from datagen import DataGenerator
from model import build_model

def compute_max_label_len(labels):
    max_len = 0
    for label in labels.values():
        if len(label) > max_len:
            max_len = len(label)
    return max_len

def main():
    partitions_dir = '/home/alex/OneDrive/phd-project/singleton-dataset-generation/dRNA/3_8_NNInputs'
    labels_file = '/home/alex/OneDrive/phd-project/singleton-dataset-generation/dRNA/3_8_NNInputs/labels.json'

    # Parameters
    batch_size = 256
    timesteps = 512          # Size of window during data preparation
    num_classes = 5          # A, C, G, U, Blank
    num_epochs = 1

    # Train/val data
    partitions = []
    for i in range(1, 11):
        with open('{0}/partition-{1}.json'.format(partitions_dir, i), 'r') as f:
            partitions.append(json.load(f))

    # Labels
    with open(labels_file, 'r') as labels_f:
        labels = json.load(labels_f)
    max_label_length = compute_max_label_len(labels)

    params = {'batch_size': batch_size,
            'window_size': timesteps,
            'max_label_len': max_label_length,
            'shuffle': True}

    # Train using k-fold CV
    cv_scores = []
    for i, partition in enumerate(partitions):
        print("Training model {0}...".format(i))

        training_generator = DataGenerator(partition['train'], labels, **params)
        validation_generator = DataGenerator(partition['val'], labels, **params)

        model = build_model(batch_size, timesteps, max_label_length, num_classes)
        model.compile(optimizer='adam', 
                      loss={'ctc': lambda labels, y_pred: y_pred})
        # For debugging:
        # model.compile(optimizer='adam', 
        #               loss={'ctc': lambda labels, y_pred: y_pred},
        #               run_eagerly=True)

        num_train_signals = len(partition['train'])
        num_val_signals = len(partition['val'])

        model.summary()
        model.fit(
            x = training_generator,
            validation_data=validation_generator,
            validation_steps = num_val_signals // batch_size,
            # validation_freq = 2,
            batch_size = batch_size,
            steps_per_epoch = num_train_signals // batch_size,
            epochs = num_epochs)
        
        score = model.evaluate(
            x = validation_generator,
            steps = num_val_signals // batch_size)
        print("Model {0} scores: Loss = {1}".format(i, score))
        cv_scores.append(score)
    print(cv_scores)

if __name__ == "__main__":
    main()
