import argparse
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.io.gfile import glob
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from textdistance import levenshtein

from data import get_dataset
from model import get_training_model, get_evaluation_model
from utilities import setup_local, get_config

# Computed elsewhere
# STEPS_PER_EPOCH = 41407

class EditDistanceCallback(Callback):
    def __init__(self, config, val_dataset):
        self.config = config
        self.val_dataset = val_dataset

    def on_epoch_end(self, logs=None):
        eval_model = get_evaluation_model(self.config, self.model.get_weights())

        distances = []
        for batch in self.val_dataset:
            inputs = batch[0]["inputs"]
            labels = batch[0]["labels"]
            input_lengths = batch[0]["input_length"]
            label_lengths = batch[0]["label_length"]

            # Pass test data into network
            softmax_out = eval_model.predict(inputs)

            # CTC decoding of network outputs
            prediction = K.ctc_decode(softmax_out, input_lengths, greedy=True, beam_width=100, top_paths=1)
            prediction = K.get_value(prediction[0][0])

            for i, pred_label in enumerate(prediction):
                signal = inputs[i]
                label = labels[i]
                label_length = label_lengths[i]
                y_pred = softmax_out[i]

                label = _to_int_list(label)
                label = _label_to_sequence(label, label_length)
                print("True label: {0}".format(label))

                pred_label = _to_int_list(pred_label)
                pred_label_len = _calculate_len_pred(pred_label)
                pred_label = _label_to_sequence(pred_label, pred_label_len)
                print("Predicted label: {0}".format(pred_label))

                edit_dist = levenshtein.normalized_distance(label, pred_label)
                print("Edit distance: {0}".format(edit_dist))
                distances.append(edit_dist)
                print("\n\n\n")
                break
        
        print("Average edit distance across test dataset: {0}".format(mean(distances)))

def train(shards_dir, checkpoint, epoch_to_resume, config_file):
    config = get_config(config_file)

    train_files = glob("/g/data/xc17/Eyras/alex/rna-basecaller/shards/debugging/CATTTTATCTCTGGGTCATT_GCCTACTTCGTCTATCACTCCT/*.tfrecords")
    train_dataset = get_dataset(train_files, config, val=False)

    val_files = glob("/g/data/xc17/Eyras/alex/rna-basecaller/shards/debugging/CATTTTATCTCTGGGTCATT_GCCTACTTCGTCTATCACTCCT/*.tfrecords")
    val_dataset = get_dataset(val_files, config, val=True)

    strategy = MirroredStrategy()
    with strategy.scope():
        model, initial_epoch = get_training_model(
            checkpoint, epoch_to_resume, config)

    edit_distance = EditDistanceCallback(config, val_dataset)

    logs_path = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(
        log_dir=logs_path, histogram_freq=1, profile_batch='500,520')

    # checkpoint_path = "model-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint_path = "model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor="val_loss", 
                                 verbose=1, 
                                 mode="min",
                                 save_weights_only=True,
                                 )
    callbacks_list = [checkpoint, tensorboard, edit_distance]

    model.summary()
    model.fit(train_dataset,
              steps_per_epoch=2000 // config.train.batch_size,
              epochs=config.train.n_epochs,
              initial_epoch=initial_epoch,
            #   validation_data=val_dataset,
            #   validation_freq=config.train.val_freq,
              verbose=1,
              callbacks=callbacks_list,
              )
    
    score = model.evaluate(x=train_dataset)
    print(score)

def train_local(checkpoint=None, initial_epoch=None):
    setup_local()
    data = '/mnt/sda/singleton-dataset-generation/dRNA/3_8_NNInputs/tfrecord_approach/shards'
    train(data, checkpoint, initial_epoch, 'config.yaml')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--checkpoint",
                        help="path to weights checkpoint to load")
    parser.add_argument("-e",
                        "--initial_epoch",
                        help="epoch to resume training at")
    parser.add_argument("-g",
                        "--config-file",
                        help="file containing model config params")
    parser.add_argument("-s",
                        "--shards-dir",
                        help="directory containing train/val/test shard files")
    args = parser.parse_args()

    if args.checkpoint is not None:
        assert args.initial_epoch is not None
        args.initial_epoch = int(args.initial_epoch)

    # train_local()
    # train_local('/home/alex/OneDrive/phd-project/rna-basecaller/train-10-local/model-57.h5', 57)

    train(args.shards_dir,
          args.checkpoint, 
          args.initial_epoch, 
          args.config_file)

    # Refs:
    # http://digital-thinking.de/keras-three-ways-to-use-custom-validation-metrics-in-keras/
    # https://github.com/cyprienruffino/CTCModel/blob/151ca5a8116ccac4a6b452e05267f9977ed76fd9/keras_ctcmodel/CTCModel.py
    # https://www.endpoint.com/blog/2019/01/08/speech-recognition-with-tensorflow
    # https://github.com/keras-team/keras/issues/7445

def _to_int_list(float_tensor):
    return K.cast(float_tensor, "int32").numpy()

def _calculate_len_pred(pred):
    for i, x in enumerate(pred):
        if x == -1:
            return i
    print("ERROR IN LENGTH PREDICTION")
    return -1

def _label_to_sequence(label, label_length):
    label = label[:label_length]
    bases = ['A', 'C', 'G', 'T']
    label = list(map(lambda b: bases[b], label))
    return "".join(label)