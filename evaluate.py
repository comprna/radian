import json

from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras import backend as K
from textdistance import levenshtein

from beam_search_decoder import ctcBeamSearch
from rna_model import RnaModel

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

def predict_beam(model, dataset, verbose=False, use_model=False):
    if use_model == True:
        # model_file = '/home/alex/Documents/rnabasecaller/6mer-probs.json'
        # model_file = '/home/150/as2781/rnabasecaller/6mer-probs.json'
        model_file = '/home/150/as2781/rnabasecaller/6mer-cond-probs.json'
        with open(model_file, 'r') as f:
            k6mer_probs = json.load(f)
        rna_model = RnaModel(k6mer_probs)
    else:
        rna_model = None

    classes = 'ACGT'
    predictions = []
    for batch in dataset:
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
            label = _label_to_sequence(label, label_length)

            # Predicted label
            pred = ctcBeamSearch(softmax_out, classes, rna_model)

            # if verbose == True:
            #     print("{}, {}".format(label, pred))

            predictions.append((label, pred))
    
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

def compute_mean_ed_beam(model, dataset, verbose=False, use_model=False):
    predictions = predict_beam(model, dataset, verbose, use_model)
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
    from tensorflow.io.gfile import glob
    from data import get_dataset
    from model import get_prediction_model
    from utilities import get_config, setup_local

    # Gadi
    config_file = '/home/150/as2781/rnabasecaller/config.yaml'
    data_dir = '/g/data/xc17/Eyras/alex/mnt-sda-backup/singleton-dataset-generation/dRNA/4_8_NNInputs/0_3_CreateTrimmedSortedTFRecords/shards/val/label_len_20'
    model_file = '/g/data/xc17/Eyras/alex/working/rna-basecaller/4_8_NNInputs/curriculum_learning/train-10/model-95.h5'

    # Local
    # setup_local()
    # config_file = '/home/alex/Documents/rnabasecaller/config.yaml'
    # data_dir = '/mnt/sda/singleton-dataset-generation/dRNA/4_8_NNInputs/0_3_CreateTrimmedSortedTFRecords/shards/val/label_len_20'
    # model_file = '/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/curriculum_learning/train-10/model-95.h5'

    config = get_config(config_file)
    test_files = glob("{}/*.tfrecords".format(data_dir))
    test_dataset = get_dataset(test_files, config.train.batch_size, val=True)
    model = get_prediction_model(model_file, config)

    # mean_ed_greedy = compute_mean_ed_greedy(model, test_dataset, verbose=True)
    # mean_ed_beam = compute_mean_ed_beam(model, test_dataset, verbose=True)
    # mean_ed_model = compute_mean_ed_beam(model, test_dataset, verbose=True, use_model=True)

    predict_all(model, test_dataset)