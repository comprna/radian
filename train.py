import argparse
import sys
import tensorflow as tf
import yaml
from attrdict import AttrDict
from data import get_batched_dataset
from datetime import datetime
from model import initialise_or_load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Computed elsewhere
STEPS_PER_EPOCH = 41407

def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))

def train(shards_dir, checkpoint, epoch_to_resume, config_file):
    c = get_config(config_file)

    train_filenames = tf.io.gfile.glob("{0}/train/*.tfrecords".format(shards_dir))
    train_dataset = get_batched_dataset(train_filenames, c, val=False)

    val_filenames = tf.io.gfile.glob("{0}/val/*.tfrecords".format(shards_dir))
    val_dataset = get_batched_dataset(val_filenames, c, val=True)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model, initial_epoch = initialise_or_load_model(checkpoint, 
                                                        epoch_to_resume, 
                                                        c)

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
              verbose = 1,
              callbacks = callbacks_list)
    
    score = model.evaluate(x = val_dataset)
    print(score)

def train_local():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    train('/mnt/sda/singleton-dataset-generation/dRNA/3_8_NNInputs/tfrecord_approach/shards', None, None, 'config.yaml')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="path to weights checkpoint to load")
    parser.add_argument("-e", "--initial_epoch", help="epoch to resume training at")
    parser.add_argument("-g", "--config-file", help="file containing model config params")
    parser.add_argument("-s", "--shards-dir", help="directory containing train/val/test shard files")
    args = parser.parse_args()

    if args.checkpoint is not None:
        assert args.initial_epoch is not None
        args.initial_epoch = int(args.initial_epoch)

    train_local()

    # train(args.shards_dir,
    #       args.checkpoint, 
    #       args.initial_epoch, 
    #       args.config_file)
