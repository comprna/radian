import cProfile
from datetime import datetime
import json
import sys

from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.io.gfile import glob
from tensorflow.keras import backend as K
from textdistance import levenshtein

from beam_search_decoder import ctcBeamSearch
from create_model import RnaModel
from data import get_dataset
from model import get_prediction_model
from rna_model_lstm import get_rna_prediction_model
from utilities import get_config, setup_local

def predict_beam(model, dataset, lm_factor, verbose=False, rna_model=None, profile=False):
    batch_n = 0
    input_n = 0
    
    # Header of tsv
    print(f"Change\tBatch\tInput\tTruth\tPred_Without\tPred_Model\tPred_Without_i\tPred_Model_i\tED_Without\tED_Model")
    classes = 'ACGT'
    n_better = 0
    n_worse = 0
    n_same = 0
    eds_without = []
    eds_model = []
    for s, batch in enumerate(dataset):
        # if s != batch_n:
        #     continue

        inputs = batch[0]["inputs"]
        labels = batch[0]["labels"]
        label_lengths = batch[0]["label_length"]

        # Pass test data into network
        softmax_out_batch = model.predict(inputs)

        # Get prediction for each input
        for i, softmax_out in enumerate(softmax_out_batch):
            # if i != input_n:
            #     continue

            # Actual label
            label = labels[i]
            label_length = label_lengths[i]
            label = _to_int_list(label)
            label_seq = _label_to_sequence(label, label_length)

            # Predicted label (without RNA model)
            pred_wout, pred_wout_i = ctcBeamSearch(softmax_out,
                                                         classes, 
                                                         None, 
                                                         label[:label_length], 
                                                         lm_factor=lm_factor)
            ed_without = levenshtein.normalized_distance(label_seq, pred_wout)
            eds_without.append(ed_without)

            # Predicted label (with RNA model)
            pred_model, pred_model_i = ctcBeamSearch(softmax_out,
                                                     classes, 
                                                     rna_model, 
                                                     label[:label_length], 
                                                     lm_factor=lm_factor)
            ed_model = levenshtein.normalized_distance(label_seq, pred_model)
            eds_model.append(ed_model)

            if verbose == True:
                if ed_model - ed_without < 0:
                    n_better += 1
                    print(f"Better\t{s}\t{i}\t{label_seq}\t{pred_wout}\t{pred_model}\t{pred_wout_i}\t{pred_model_i}\t{ed_without}\t{ed_model}")

                elif ed_model - ed_without > 0:
                    n_worse += 1
                    print(f"Worse\t{s}\t{i}\t{label_seq}\t{pred_wout}\t{pred_model}\t{pred_wout_i}\t{pred_model_i}\t{ed_without}\t{ed_model}")

                else:
                    n_same += 1
                    print(f"Same\t{s}\t{i}\t{label_seq}\t{pred_wout}\t{pred_model}\t{pred_wout_i}\t{pred_model_i}\t{ed_without}\t{ed_model}")

            if profile == True:
                return

            # plt.imshow(np.transpose(softmax_out), cmap="gray_r", aspect="auto")
            # plt.show()

            plot_softmax(inputs[i], softmax_out, label_seq, pred_model, pred_model_i, pred_wout, pred_wout_i, s, i)

    print(f"# better: {n_better}")
    print(f"# worse:  {n_worse}")
    print(f"# same:   {n_same}")
    print(f"% better: {n_better/(n_better+n_worse+n_same)*100}")
    print(f"% worse:  {n_worse/(n_better+n_worse+n_same)*100}")
    print(f"% same:   {n_same/(n_better+n_worse+n_same)*100}")

    print(f"Average ED without RNA model: {mean(eds_without)}")
    print(f"Average ED with RNA model: {mean(eds_model)}")

def plot_softmax(signal, matrix, actual, pred_model, pred_model_i, pred_wout, pred_wout_i, batch_n, input_n):
    # Display timesteps horizontally rather than vertically
    t_matrix = np.transpose(matrix)

    # Share time axis to allow for comparison
    fig, axs = plt.subplots(3, 1, sharex="all", figsize=(20,10))

    # Plot signal
    axs[0].set_title("Raw Signal")
    axs[0].plot(signal)

    # Plot spikes
    axs[1].set_title("CTC Output (spikes)")
    axs[1].plot(t_matrix[4], label="b", color="grey", linestyle="dashed")
    axs[1].plot(t_matrix[0], label="A", color="red")
    axs[1].plot(t_matrix[1], label="C", color="orange")
    axs[1].plot(t_matrix[2], label="G", color="green")
    axs[1].plot(t_matrix[3], label="T", color="blue")
    axs[1].legend()

    # Plot probability matrix
    axs[2].set_title("CTC Output (shaded)")
    grid = axs[2].imshow(t_matrix, cmap="gray_r", aspect="auto")

    overlay_prediction(axs[2], pred_model, pred_model_i, "orange", offset=0.5)
    overlay_prediction(axs[2], pred_wout, pred_wout_i, "blue")

    ed_model = levenshtein.normalized_distance(actual, pred_model)
    ed_without = levenshtein.normalized_distance(actual, pred_wout)

    if ed_model - ed_without < 0:
        change = "better"
    elif ed_model - ed_without > 0:
        change = "worse"
    else:
        change = "same"

    axs[2].text(0, 5.5, "GT", fontsize="x-large", color="green")
    axs[2].text(0, 6, "-M", fontsize="x-large", color="blue")
    axs[2].text(0, 6.5, "+M", fontsize="x-large", color="orange")
    axs[2].text(50, 5.5, actual, fontsize="x-large", color="green")
    axs[2].text(50, 6, pred_wout, fontsize="x-large", color="blue")
    axs[2].text(50, 6.5, pred_model, fontsize="x-large", color="orange")
    axs[2].text(600, 6, f"{ed_without:.5f}", fontsize='x-large', color="blue")
    axs[2].text(600, 6.5, f"{ed_model:.5f} ({change})", fontsize='x-large', color="orange")

    fig.suptitle(f"Ground truth: {actual}")
    plt.savefig(f"{batch_n}-{input_n}-{change}.png", bbox_inches="tight", pad_inches=0)
    plt.show()

def overlay_prediction(plot, prediction, indices, color, offset=0):
    bases = ['A', 'C', 'G', 'T']
    for i, nt in enumerate(prediction):
        plot.text(indices[i], bases.index(nt)+offset, nt,  fontsize='x-large', color=color)

def compute_mean_ed_beam(model, dataset, verbose=False, rna_model=None):
    print(f"Change\tBatch\tInput\tTruth\tPred_Without\tPred_Model\tPred_Without_i\tPred_Model_i\tED_Without\tED_Model")
    predict_beam(model, dataset, verbose, rna_model)

def compute_mean_ed(predictions, verbose=False):
    eds = []
    for p in predictions:
        ed = levenshtein.normalized_distance(p[0], p[1])
        if verbose == True:
            print('{}, {}, {}'.format(p[0], p[1], ed))
        eds.append(ed)

    return mean(eds)

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

def callback():
    # # Gadi
    # s_config_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/with-rna-model/train-3-37/s-config-3.yaml'
    # r_config_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/with-rna-model/train-3-37/r-config-37.yaml'
    # data_dir = '/g/data/xc17/Eyras/alex/working/2_0_8_WriteTFRecords/3/1024_128/val'
    # s_model_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/train-3/model-10.h5'
    # r_model_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/with-rna-model/train-3-37/r-train-37-model-03.h5'
    lm_factor = 0.5

    # # Local
    setup_local()
    s_config_file = '/home/alex/OneDrive/phd-project/rna-basecaller/experiments/with-rna-model/train-3-37/s-config-3.yaml'
    r_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/r-config-37.yaml'
    data_dir = '/mnt/sda/basecaller-data/dRNA/2_ProcessTrainingData/0_8_WriteTFRecords/3/1024_128/val'
    s_model_file = '/mnt/sda/rna-basecaller/experiments/sig-to-seq/dRNA/train-3/model-10.h5'
    r_model_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/r-train-37-model-03.h5'

    s_config = get_config(s_config_file)
    r_config = get_config(r_config_file)

    test_files = glob("{}/*.tfrecords".format(data_dir))
    test_dataset = get_dataset(test_files, s_config.train.batch_size, val=True)

    s_model = get_prediction_model(s_model_file, s_config)
    r_model = get_rna_prediction_model(r_model_file, r_config)

    predict_beam(s_model, test_dataset, lm_factor, verbose=True, rna_model=r_model, profile=True)

if __name__ == "__main__":

    # cProfile.run('callback()', sort='cumtime')

    # # Gadi
    # s_config_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/with-rna-model/train-3-37/s-config-3.yaml'
    # r_config_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/with-rna-model/train-3-37/r-config-37.yaml'
    # data_dir = '/g/data/xc17/Eyras/alex/working/2_0_8_WriteTFRecords/3/1024_128/val'
    # s_model_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/train-3/model-10.h5'
    # r_model_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/with-rna-model/train-3-37/r-train-37-model-03.h5'
    # lm_factor = float(sys.argv[1])

    # Local
    setup_local()
    s_config_file = '/home/alex/OneDrive/phd-project/rna-basecaller/experiments/with-rna-model/train-3-37/s-config-3.yaml'
    r_config_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/r-config-37.yaml'
    data_dir = '/mnt/sda/basecaller-data/dRNA/2_ProcessTrainingData/0_8_WriteTFRecords/3/1024_128/val'
    s_model_file = '/mnt/sda/rna-basecaller/experiments/sig-to-seq/dRNA/train-3/model-10.h5'
    r_model_file = '/mnt/sda/rna-basecaller/experiments/with-rna-model/train-3-37/r-train-37-model-03.h5'
    lm_factor = 0.5

    s_config = get_config(s_config_file)
    r_config = get_config(r_config_file)

    test_files = glob("{}/*.tfrecords".format(data_dir))
    test_dataset = get_dataset(test_files, s_config.train.batch_size, val=True)

    s_model = get_prediction_model(s_model_file, s_config)
    r_model = get_rna_prediction_model(r_model_file, r_config)

    # test_x = tf.one_hot([0,0,1,0,0,1,3,0], 4).numpy()
    # test_x = test_x.reshape(1,8,4)
    # print(r_model.predict(test_x))

    predict_beam(s_model, test_dataset, lm_factor, verbose=True, rna_model=r_model)