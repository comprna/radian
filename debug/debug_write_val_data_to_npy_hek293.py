import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.io.gfile import glob
from tensorflow.keras import backend as K

from data import get_dataset
from model import get_prediction_model
from utilities import get_config, setup_local

N_WINDOWS_PER_SHARD = 50000

def label_to_sequence(label, label_length):
    label = K.cast(label, "int32").numpy()
    label = label[:label_length]
    bases = ['A', 'C', 'G', 'T']
    label = list(map(lambda b: bases[b], label))
    return "".join(label)

def main():
    setup_local()

    # Config (incl. batch size and window length)
    s_config_file = '/home/alex/OneDrive/phd-project/rna-basecaller/experiments/with-rna-model/train-3-37/s-config-3.yaml'
    s_config = get_config(s_config_file)

    s_model_file = '/mnt/sda/rna-basecaller/experiments/sig-to-seq/dRNA/train-3/model-10.h5'
    s_model = get_prediction_model(s_model_file, s_config)

    # Val TFRecords
    data_dir = '/mnt/sda/basecaller-data/dRNA/2_ProcessTrainingData/0_8_WriteTFRecords/3/1024_128/val'
    val_files = glob("{}/*.tfrecords".format(data_dir))
    val_files = [f"{data_dir}/hek293-3-1024-128-shard-11.tfrecords",
                 f"{data_dir}/hek293-3-1024-128-shard-17.tfrecords",
                 f"{data_dir}/hek293-3-1024-128-shard-5.tfrecords",
                 f"{data_dir}/hek293-3-1024-128-shard-8.tfrecords"]

    # Val metadata
    val_info_file = '/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/3_Get_Reads_Per_Val_Shard/hek293_1024_128_val_data.tsv'
    val_info = pd.read_csv(val_info_file, sep='\t')

    # Reads to exclude
    exclude_file = '/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/3_Get_Reads_Per_Val_Shard/hek293_1024_128_val_reads_to_exclude.txt'
    with open(exclude_file, "r") as f:
        exclude_reads = [line.rstrip('\n') for line in f]

    # Iterate through val files & store windows per read
    read_windows_all = []
    read_labels_all = []
    read_ids = []
    for file in val_files:

        # Get shard info
        shard = int(file.split("-")[-1].split(".")[0])
        shard_info = val_info[val_info['shard_id'] == shard]
        print(shard_info)

        # Get ready to read shard
        shard_dataset = get_dataset([file], s_config.train.batch_size, val=True)

        # Iterate through shard
        shard_pos = 1
        curr_read_windows = []
        curr_read_labels = []
        curr_read_id = shard_info[shard_info['shard_pos'] == shard_pos].read.squeeze()
        for b, batch in enumerate(shard_dataset):
            inputs = batch[0]["inputs"]
            labels = batch[0]["labels"]
            label_lengths = batch[0]["label_length"]

            for j, _ in enumerate(inputs):

                # Skip over this window if it is part of an exclude read
                # (one split over multiple shards and therefore not
                # fully contained in the val dataset).
                skip_window = curr_read_id in exclude_reads

                if not skip_window:
 
                    # Match current window with metadata info
                    label = label_to_sequence(labels[j], label_lengths[j])
                    expected_label = shard_info[shard_info['shard_pos'] == shard_pos].window_label.squeeze()

                    # Store signal and label for current window in read
                    if label != expected_label:
                        print("ERROR! Labels do not match up")
                    else:
                        # Verify softmax matches label
                        print(label)
                        print(expected_label)

                        # Compute softmax for current window
                        softmax = s_model.predict(np.array([inputs[j].numpy(),]))
                        softmax = np.squeeze(softmax)
                        plt.imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
                        plt.show()


                        curr_read_windows.append(inputs[j].numpy())
                        curr_read_labels.append(expected_label)

                # Move to next window
                shard_pos += 1

                # If we are up to the next read in the shard then
                # initialise it.
                if shard_pos <= N_WINDOWS_PER_SHARD:
                    read_pos = shard_info[shard_info['shard_pos'] == shard_pos].read_pos.squeeze()
                    if read_pos == 1 and curr_read_id not in exclude_reads:
                        # First store the old read if isn't a skip read
                        read_windows_all.append(curr_read_windows)
                        read_labels_all.append(curr_read_labels)
                        read_ids.append(curr_read_id)
                        # Then get ready to store new read
                        curr_read_windows = []
                        curr_read_labels = []
                    curr_read_id = shard_info[shard_info['shard_pos'] == shard_pos].read.squeeze()
                    


    # Write all data to file
    with open("read_windows_all.npy", "wb") as f:
        np.save(f, read_windows_all)
    with open("read_labels_all.npy", "wb") as f:
        np.save(f, read_labels_all)
    with open("read_ids.npy", "wb") as f:
        np.save(f, read_ids)

if __name__ == "__main__":
    main()
