import argparse
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.io.gfile import glob
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from data import get_dataset
from model import get_training_model
from utilities import setup_local, get_config


# Computed elsewhere
STEPS_PER_EPOCH = 41407

def train(shards_dir, checkpoint, epoch_to_resume, config_file):
    config = get_config(config_file)

    train_files = glob("{0}/train/*.tfrecords".format(shards_dir))
    train_dataset = get_dataset(train_files, config, val=False)

    val_files = glob("{0}/val/*.tfrecords".format(shards_dir))
    val_dataset = get_dataset(val_files, config, val=True)

    strategy = MirroredStrategy()
    with strategy.scope():
        model, initial_epoch = get_training_model(
            checkpoint, epoch_to_resume, config)

    logs_path = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(
        log_dir=logs_path, histogram_freq=1, profile_batch='500,520')

    checkpoint_path = "model-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor="val_loss", 
                                 verbose=1, 
                                 mode="min",
                                 save_weights_only=True,
                                 )
    callbacks_list = [checkpoint, tensorboard]

    model.summary()
    model.fit(train_dataset,
              steps_per_epoch=STEPS_PER_EPOCH,
              epochs=config.train.n_epochs,
              initial_epoch=initial_epoch,
              validation_data=val_dataset,
              validation_freq=config.train.val_freq,
              verbose=1,
              callbacks=callbacks_list,
              )
    
    score = model.evaluate(x=val_dataset)
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
    # train_local('test-checkpoint/model-06-27.44.h5', 6)

    train(args.shards_dir,
          args.checkpoint, 
          args.initial_epoch, 
          args.config_file)
