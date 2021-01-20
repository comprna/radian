import json

from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import backend as K
from textdistance import levenshtein

from beam_search_decoder import ctcBeamSearch
from rna_model import RnaModel

# def plot_softmax_output(matrix, label):
#     print(matrix.shape)
#     t_matrix = np.transpose(matrix)
#     plt.imshow(t_matrix, cmap="gray_r", aspect="auto")
#     plt.show()
#     plt.savefig('softmax_outputs/{0}'.format(label))

#     # TODO: Why is there not a value for all 512 timesteps in the softmax output?

def compute_mean_edit_distance(model, dataset, verbose=False):
    classes = 'ACGT'

    with open('6mer-probs.json', 'r') as f:
        k6mer_probs = json.load(f)
    rna_model = RnaModel(k6mer_probs)

    greedy_distances = []
    beam_distances = []
    model_distances = []

    for batch in dataset:
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

        for i, softmax_out in enumerate(softmax_out_batch):
            label = labels[i]
            label_length = label_lengths[i]
            label = _to_int_list(label)
            label = _label_to_sequence(label, label_length)
            print("True label: {0}".format(label))

            # plot_softmax_output(softmax_out, label)

            greedy_pred = _to_int_list(greedy_pred_batch[i])
            greedy_pred_len = _calculate_len_pred(greedy_pred)
            greedy_pred = _label_to_sequence(greedy_pred, greedy_pred_len)
            greedy_edit_dist = levenshtein.normalized_distance(label, greedy_pred)
            greedy_distances.append(greedy_edit_dist)
            print("Predicted label (greedy): {0}".format(greedy_pred))

            beam_pred = ctcBeamSearch(softmax_out, classes, None)
            beam_edit_dist = levenshtein.normalized_distance(label, beam_pred)
            beam_distances.append(beam_edit_dist)

            model_pred = ctcBeamSearch(softmax_out, classes, rna_model)
            model_edit_dist = levenshtein.normalized_distance(label, model_pred)
            model_distances.append(model_edit_dist)

            if verbose == True:
                print("True label: {0}".format(label))
                print("Predicted label (greedy): {0}".format(greedy_pred))
                print("Edit distance (greedy): {0}".format(greedy_edit_dist))
                print("Predicted label (beam): {0}".format(beam_pred))
                print("Edit distance (beam): {0}".format(beam_edit_dist))
                print("Predicted label (model): {0}".format(model_pred))
                print("Edit distance (model): {0}\n\n\n".format(model_edit_dist))

    print("Mean ed greedy: {0}".format(mean(greedy_distances)))
    print("Mean ed beam: {0}".format(mean(beam_distances)))
    print("Mean ed beam with model: {0}".format(mean(model_distances)))

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