import tensorflow as tf
import tcn
from datagen import DataGenerator
from model import build_model

def main():
    # Data info
    max_label_length = 100   # Computed during data preparation
    num_train_signals = 1000 # Total number of windows in training set
    num_test_signals = 1000

    # Parameters
    batch_size = 200
    timesteps = 512          # Size of window during data preparation
    num_classes = 5          # A, C, G, U, Blank
    num_epochs = 2

    train_csv_file = '/mnt/sda/hek293-fold1.csv'
    train_gen = DataGenerator(train_csv_file, batch_size, timesteps, False, max_label_length)

    test_csv_file = '/mnt/sda/hek293-fold1.csv'
    test_gen = DataGenerator(test_csv_file, batch_size, timesteps, False, max_label_length)

    model = build_model(batch_size, timesteps, max_label_length, num_classes)
    model.compile(optimizer='adam', loss={'ctc': lambda labels, y_pred: y_pred})

    tcn.tcn_full_summary(model, expand_residual_blocks=False)
    model.fit(
        x = train_gen.next_batch(),
        steps_per_epoch = num_train_signals // batch_size,
        validation_data = test_gen.next_batch(),
        validation_steps = num_test_signals // batch_size,
        epochs = num_epochs)

if __name__ == "__main__":
    main()
