import cProfile
from datetime import datetime
import json

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
from rna_model import get_rna_prediction_model
from utilities import get_config, setup_local

def predict_greedy(model, dataset, verbose=False, plot=False, model_id=None):
    predictions = []
    for s, batch in enumerate(dataset):
        inputs = batch[0]["inputs"]
        labels = batch[0]["labels"]
        input_lengths = batch[0]["input_length"]
        label_lengths = batch[0]["label_length"]

        # Pass test data into network
        softmax_out_batch = model.predict(inputs)

        # Greedy decoding
        greedy_pred_batch = K.ctc_decode(softmax_out_batch,
                                  input_lengths,
                                  greedy=True,
                                  beam_width=100,
                                  top_paths=1)
        greedy_pred_batch = K.get_value(greedy_pred_batch[0][0])

        # Get prediction for each input
        for i, softmax_out in enumerate(softmax_out_batch):
            # Actual label
            label = labels[i]
            label_length = label_lengths[i]
            label = _to_int_list(label)
            label = _label_to_sequence(label, label_length)

            # Predicted label
            greedy_pred = _to_int_list(greedy_pred_batch[i])
            greedy_pred_len = _calculate_len_pred(greedy_pred)
            greedy_pred = _label_to_sequence(greedy_pred, greedy_pred_len)

            # Plot the signal and prediction for debugging
            if plot == True:
                plot_softmax(inputs[i], softmax_out, label, greedy_pred, model_id, i)
            # if verbose == True:
            #     print("{}, {}".format(label, greedy_pred))

            predictions.append((label, greedy_pred))
    
            # If we are in plotting mode, only plot the first batch
            if plot == True and i == 5:
                return

    return predictions

def predict_beam(model, dataset, verbose=False, rna_model=None):
    classes = 'ACGT'
    predictions = []
    for s, batch in enumerate(dataset):
        inputs = batch[0]["inputs"]
        labels = batch[0]["labels"]
        label_lengths = batch[0]["label_length"]

        # Pass test data into network
        softmax_out_batch = model.predict(inputs)

        # Get prediction for each input
        for i, softmax_out in enumerate(softmax_out_batch):
            # Actual label
            label = labels[i]
            label_length = label_lengths[i]
            label = _to_int_list(label)
            label_seq = _label_to_sequence(label, label_length)

            # Predicted label (without RNA model)
            start = datetime.now()
            pred_without = ctcBeamSearch(softmax_out, classes, None, label[:label_length])
            end = datetime.now()
            time_without = end - start
            ed_without = levenshtein.normalized_distance(label_seq, pred_without)

            # Predicted label (with RNA model)
            start = datetime.now()
            pred_model = ctcBeamSearch(softmax_out, classes, rna_model, label[:label_length])
            end = datetime.now()
            time_with = end - start
            ed_model = levenshtein.normalized_distance(label_seq, pred_model)

            if verbose == True:
                print(f"Truth: {label_seq}")
                print(f"W/O:   {pred_without}")
                print(f"With:  {pred_model}")
                print(f"ED w/o:  {ed_without}\nED with: {ed_model}\n\n")
                if ed_model - ed_without < 0:
                    print(f"MODEL IMPROVES ACCURACY !!!!! Batch {s} Input {i}\n\n")
                elif ed_model - ed_without > 0:
                    print(f"Model worsens accuracy :( Batch {s} Input {i}\n\n")

            # plt.imshow(np.transpose(softmax_out), cmap="gray_r", aspect="auto")
            # plt.show()

            predictions.append((label_seq, pred_model))

    return predictions

def predict_all(model, dataset, verbose=False, plot=False, model_id=None):
    # model_file = '/home/alex/Documents/rnabasecaller/6mer-probs.json'
    # model_file = '/home/150/as2781/rnabasecaller/6mer-probs.json
    model_file = '/home/150/as2781/rnabasecaller/6mer-cond-probs.json'
    with open(model_file, 'r') as f:
        k6mer_probs = json.load(f)
    rna_model = RnaModel(k6mer_probs)

    classes = 'ACGT'
    predictions = []
    for s, batch in enumerate(dataset):
        inputs = batch[0]["inputs"]
        labels = batch[0]["labels"]
        input_lengths = batch[0]["input_length"]
        label_lengths = batch[0]["label_length"]

        # Pass test data into network
        softmax_out_batch = model.predict(inputs)

        # Greedy decoding
        greedy_pred_batch = K.ctc_decode(softmax_out_batch,
                                  input_lengths,
                                  greedy=True,
                                  beam_width=100,
                                  top_paths=1)
        greedy_pred_batch = K.get_value(greedy_pred_batch[0][0])

        # Get prediction for each input
        for i, softmax_out in enumerate(softmax_out_batch):
            # Actual label
            label = labels[i]
            label_length = label_lengths[i]
            label = _to_int_list(label)
            label = _label_to_sequence(label, label_length)

            # Greedy prediction
            greedy_pred = _to_int_list(greedy_pred_batch[i])
            greedy_pred_len = _calculate_len_pred(greedy_pred)
            greedy_pred = _label_to_sequence(greedy_pred, greedy_pred_len)
            greedy_ed = levenshtein.normalized_distance(greedy_pred, label)

            # Beam search prediction - no model
            beam_pred = ctcBeamSearch(softmax_out, classes, None)
            beam_ed = levenshtein.normalized_distance(beam_pred, label)

            # Beam search prediction - with model
            model_pred = ctcBeamSearch(softmax_out, classes, rna_model)
            model_ed = levenshtein.normalized_distance(model_pred, label)

            print("{}, {}, {}, {}, {}, {}, {}".format(label, greedy_pred, 
                greedy_ed, beam_pred, beam_ed, model_pred, model_ed))


def plot_softmax(signal, matrix, actual, predicted, model_id, data_id):
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

    # Get edit distance
    ed = levenshtein.normalized_distance(actual, predicted)

    fig.suptitle("Actual: {}   Predicted: {}   ED: {}".format(actual, predicted, ed))
    plt.savefig("{}-{}.png".format(model_id, data_id))

def compute_mean_ed_beam(model, dataset, verbose=False, rna_model=None):
    predictions = predict_beam(model, dataset, verbose, rna_model)
    return compute_mean_ed(predictions, verbose)

def compute_mean_ed_greedy(model, dataset, verbose=False):
    predictions = predict_greedy(model, dataset, verbose)
    return compute_mean_ed(predictions, verbose)

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

if __name__ == "__main__":

    # # Gadi
    # s_config_file = '/home/150/as2781/rnabasecaller/config.yaml'
    # data_dir = '/g/data/xc17/Eyras/alex/mnt-sda-backup/singleton-dataset-generation/dRNA/4_8_NNInputs/0_3_CreateTrimmedSortedTFRecords/shards/val/label_len_20'
    # model_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/4_8_NNInputs/curriculum_learning/train-10/model-95.h5'

    # Local
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

    test_x = tf.one_hot([0,0,1,0,0,1,3,0], 4).numpy()
    test_x = test_x.reshape(1,8,4)
    print(r_model.predict(test_x))

    # mean_ed_greedy = compute_mean_ed_greedy(model, test_dataset, verbose=True)
    mean_ed_beam = compute_mean_ed_beam(s_model, test_dataset, verbose=True, rna_model=r_model) # Pass RNA model here!
    # mean_ed_model = compute_mean_ed_beam(model, test_dataset, verbose=True, use_model=True)

    # predict_all(model, test_dataset)