import tensorflow as tf
import tcn
from datagen import DataGenerator
from model import build_model

def main():
    # Data info
    max_label_length = 39   # Computed during data preparation
    num_train_signals = 166138 # Total number of windows in training set
    num_val_signals = 166138

    # Parameters
    batch_size = 2
    timesteps = 512          # Size of window during data preparation
    num_classes = 5          # A, C, G, U, Blank
    num_epochs = 2

    params = {'batch_size': batch_size,
              'window_size': timesteps,
              'max_label_len': max_label_length,
              'shuffle': True}

    partition = {'train': ['HEK-1-1', 'HEK-1-2', 'HEK-1-3'], 'validation': ['HEK-1-4']}
    labels = {'HEK-1-1': ['A', 'C'], 'HEK-1-2': ['A', 'C', 'G'], 'HEK-1-3': ['T', 'C', 'G'], 'HEK-1-4': ['C', 'C', 'T']}

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    model = build_model(batch_size, timesteps, max_label_length, num_classes)
    model.compile(optimizer='adam', loss={'ctc': lambda labels, y_pred: y_pred}, run_eagerly=True)

    model.summary()
    model.fit(
        x = training_generator,
        batch_size = batch_size,
        steps_per_epoch = num_train_signals // batch_size,
        epochs = num_epochs)

if __name__ == "__main__":
    main()
