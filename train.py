import argparse
from datetime import datetime
import json
import os
import sys

import tensorflow as tf
from tensorflow.distribute.experimental import MultiWorkerMirroredStrategy
from tensorflow.io.gfile import glob
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback

from data import get_dataset
from evaluate import compute_mean_ed_greedy
from model import get_training_model, get_evaluation_model
from utilities import setup_local, get_config

print("Finished imports")

# Computed in utilities.py : count_n_steps_per_epoch()

STEPS_PER_EPOCH = 911506 # 3 / 256 / 32 / Batch size 32
# STEPS_PER_EPOCH = 455502 # 3 / 256 / 64 / Batch size 32
# STEPS_PER_EPOCH = 233880 # 3 / 512 / 128 / Batch size 32
# STEPS_PER_EPOCH = 466686 # 3 / 512 / 64 / Batch size 32
# STEPS_PER_EPOCH = 229070 # 3 / 1024 / 128 / Batch size 32
# STEPS_PER_EPOCH = 458635 # 3 / 1024 / 64 / Batch size 32

# An edit distance metric cannot be created since the model during training
# only outputs the loss, whereas we need the softmax matrix to compute
# edit distance.

# Perhaps we could specify the train/eval model using a training argument
# in the call function, to allow a metric: https://keras.io/api/models/model/
class EditDistanceCallback(Callback):
    def __init__(self, config, train_dataset, val_dataset, interval=10):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0 and epoch != 0:
            eval_model = get_evaluation_model(self.config, self.model.get_weights())
            # train_ed = compute_mean_ed_greedy(eval_model, self.train_dataset, verbose=True)
            val_ed = compute_mean_ed_greedy(eval_model, self.val_dataset, verbose=True)
            # print("Mean ED (train) greedy: {0}".format(train_ed))
            print("Mean ED (val) greedy: {0}".format(val_ed))
            # tf.summary.scalar('edit distance (train) greedy', data=train_ed, step=epoch)
            tf.summary.scalar('edit distance (val) greedy', data=val_ed, step=epoch)

def train(shards_dir, checkpoint, epoch_to_resume, config_file, strategy):
    print("Inside train function...")
    config = get_config(config_file)

    print("Creating datasets...")
    train_files = glob("{}/train/*.tfrecords".format(shards_dir))
    train_dataset = get_dataset(train_files, config.train.batch_size, val=False)
    train_dataset_for_eval = get_dataset(train_files, config.train.batch_size, val=True)

    val_files = glob("{}/val/*.tfrecords".format(shards_dir))
    val_dataset = get_dataset(val_files, config.train.batch_size, val=True)

    print("Creating model...")
    with strategy.scope():
        model, initial_epoch = get_training_model(
            checkpoint, epoch_to_resume, config)
    
    print("Setting up callbacks...")
    logs_path = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logs_path + "/metrics")
    file_writer.set_as_default()
    tensorboard = TensorBoard(log_dir=logs_path,
                              histogram_freq=1,
                              # write_grads=True, # Currently deprecated: https://github.com/tensorflow/tensorflow/issues/31173
                              )

    edit_distance = EditDistanceCallback(config, train_dataset_for_eval, val_dataset)

    checkpoint_path = "model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor="val_loss", 
                                 verbose=1, 
                                 mode="min",
                                 save_weights_only=True,
                                 )
    callbacks_list = [checkpoint, tensorboard, edit_distance]

    print("About to start training...")
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
    
    score = model.evaluate(x=train_dataset)
    print(score)

# def train_local(checkpoint=None, initial_epoch=None):
#     setup_local()
#     shards_dir = '/mnt/sda/singleton-dataset-generation/dRNA/4_8_NNInputs/0_2_CreateTFRecords/2_WriteTFRecords/shards/local_testing'
#     train(shards_dir, checkpoint, initial_epoch, 'config.yaml')

if __name__ == "__main__":
    print("Top of main...")
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

    print("Setting TF CONFIG...")

    # Set TF_CONFIG for MultiWorkerMirroredStrategy
    with open('tensorflow_nodefile','r') as fid:
        workers = [ host+':12345' for host in fid]

    host=os.uname()[1]
    idx = workers.index(host)
    config_json = {'cluster': { 'worker': workers }, 'task': {'type': 'worker', 'index': idx} }
    os.environ["TF_CONFIG"] = json.dumps(config_json)

    print("Setting up strategy...")

    strategy = MultiWorkerMirroredStrategy()

    print("About to start training...")

    train(args.shards_dir,
          args.checkpoint, 
          args.initial_epoch, 
          args.config_file,
          strategy)
