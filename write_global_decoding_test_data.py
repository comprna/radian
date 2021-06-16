import json

from matplotlib import pyplot as plt
import numpy as np
from tensorflow.io.gfile import glob
from tensorflow.keras import backend as K

from data import get_dataset
from model import get_prediction_model
from utilities import get_config, setup_local

STEP_SIZE = 128
WINDOW_LEN = 1024
OVERLAP = WINDOW_LEN-STEP_SIZE

def reconstruct(orig_signal, new_signal):
    reconstructed = []
    # Only reconstruct if the windows are consecutive parts of the same signal
    if np.array_equal(orig_signal[-1*OVERLAP:], new_signal[:OVERLAP]):
        reconstructed = np.concatenate((orig_signal, new_signal[-1*STEP_SIZE:]))
    return reconstructed

def label_to_sequence(label, label_length):
    label = K.cast(label, "int32").numpy()
    label = label[:label_length]
    bases = ['A', 'C', 'G', 'T']
    label = list(map(lambda b: bases[b], label))
    return "".join(label)

def main():
    setup_local()
    s_config_file = '/home/alex/OneDrive/phd-project/rna-basecaller/experiments/with-rna-model/train-3-37/s-config-3.yaml'
    s_model_file = '/mnt/sda/rna-basecaller/experiments/sig-to-seq/dRNA/train-3/model-10.h5'
    data_dir = '/mnt/sda/basecaller-data/dRNA/2_ProcessTrainingData/0_8_WriteTFRecords/3/1024_128/val'

    s_config = get_config(s_config_file)
    s_model = get_prediction_model(s_model_file, s_config)

    test_files = glob("{}/*.tfrecords".format(data_dir))
    test_dataset = get_dataset(test_files, s_config.train.batch_size, val=True)

    orig_signals = []
    windows_all = []
    window_labels_all = []
    for b, batch in enumerate(test_dataset):
        # Only deal with every 10 batches so that we reconstruct
        # signals from different reads
        if b % 5 != 0:
            continue
        inputs = batch[0]["inputs"]
        labels = batch[0]["labels"]
        label_lengths = batch[0]["label_length"]

        orig_signal = inputs[0].numpy()
        windows = [inputs[0].numpy()]
        window_labels = [label_to_sequence(labels[0], label_lengths[0])]
        for i, signal in enumerate(inputs):
            # Already processed 0th item before loop
            if i == 0:
                continue
            
            # Only reconstruct 30 windows
            if i >= 30:
                break

            orig_signal = reconstruct(orig_signal, signal.numpy())
            windows.append(signal.numpy())
            window_labels.append(label_to_sequence(labels[i], label_lengths[i]))

            # If consecutive windows cannot be recovered then move
            # onto next batch
            if len(orig_signal) == 0:
                break

        # If original signal has been recovered from 30 windows of 
        # length 1024 step size 128, it should be this long
        expected_length = 1024 + 128*29

        if len(orig_signal) == expected_length:
            orig_signals.append(orig_signal)
            windows_all.append(windows)
            window_labels_all.append(window_labels)

        if len(orig_signals) == 10:
            break

    ## Visually inspect to verify signal reconstruction
    # for i, orig_signal in enumerate(orig_signals):
    #     fig, axs = plt.subplots(30, 1, sharex="all")
    #     for j in range(30):
    #         axs[j].plot(windows_all[i][j])
    #     plt.show()
    #     plt.plot(orig_signal)
    #     plt.show()

    # # Print window labels then reconstruct complete GT labels manually
    # for window_labels in window_labels_all:
    #     print("\n\nNEXT SET OF LABELS...")
    #     for label in window_labels:
    #         print(label)

    # GT labels (computed manually elsewhere)
    gt_labels = ["AACAACTATACCTGAGATCTTATCCTAACGCGACAATAGGGATCCCATTGAACAAGGCAACCAGTTCAATAACCTAGTTAACTCATATCATCAAGCGAAACTGACCACTTCAGAATCGTACATGACGAGCCTCCAACCCAAGACGAGGCTCCAGCGGGGTTGGC", "AGCCCTACAGGACTAGGTTGTAGCTCCAGCATTTGGGATAACAACTATACCTGAGATCTTATCCTAACGCGACAATAGGGATCCCATTGAACAAGGCAACCAGTTCAATAACCTAGTTAACTCATATCATCAAGCGAAACTGACCACTTCAGAAT", "GCACAATGTAGCGCGGTAGTAACCATATACCAATCACACAACCAATCATCCGGATCATACTCCTCGCAATACCTCACCTTCACTTTAGTGTACCGATCCGGCCTCCAGTAATCCTCCCGACTCTCCCGGGGACAA", "ACATTAGGTGTAAGTGACTCATCTTGAACATAACTAGTAACCCTGGGTCAAACAAGGTCCCGAGACCCAATAAAGACAGGGTTGTTTGTAGACCTAACTTGTTACGGTCTGCGTTCTCTAT", "CCCAATTAGCACACTGGCGCCACCGACCGTGCTTTAACTGGTTGGGACCCCAATCATATCGAATCAATTTGAAAGCAAATAACGATTTCCAATTAGTGACGACAAAGGGCA", "CACGTACTCATCCACCGGACGTCATTACAATCGCCAATCCGCATGCCGGTC", "CTTACATCTATCTTTGGCTGGACCTAATGAGGCCAGACTTGAGTCTAGTGCATCCTGAAATTAGCAACTTGTTTGCTTGGAAATTATCGCCGACGTGGTAGCCCTACAGGACTAGGTTG", "GACTCGCTACCAGTGAAACGGGTCGTCGAACAAATTGAGGAGCAGCAACGCCTACCGCTCGACGTCCACCGCCCCTTAATAGGACCAGAAGAACAACAGTGCGCGC", "TTTAGACCAATCACTACTGGTCCTCGTCCGCGACTTCGAAAGTTTCGGAATGAAGAAAATAGTCGTCGGGCCTAAAATATTTGCTACTTCG", "TAGTCGTCGTGAAGGTCGAGGAACTGCAACACCTGGTCCTTGAAGGCCTTCGGTGACCCGTCGTACACGAAACAAAAAAACAACGAAGGTATTGGTTACAACCCGTAGT"]

    # Compute softmax for every window
    softmaxes_all = []
    for windows in windows_all:
        softmaxes = []
        for window in windows:
            softmax = s_model.predict(np.array([window,]))
            softmax = np.squeeze(softmax)
            softmaxes.append(softmax)
        softmaxes_all.append(softmaxes)

    # Write all data to file
    with open("orig_signals.npy", "wb") as f:
        np.save(f, orig_signals)
    with open("gt_labels.npy", "wb") as f:
        np.save(f, gt_labels)
    with open("windows_all.npy", "wb") as f:
        np.save(f, windows_all)
    with open("window_labels_all.npy", "wb") as f:
        np.save(f, window_labels_all)
    with open("softmaxes_all.npy", "wb") as f:
        np.save(f, softmaxes_all)


if __name__ == "__main__":
    main()