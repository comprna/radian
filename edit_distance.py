from statistics import mean

from tensorflow.keras import backend as K
from textdistance import levenshtein

def greedy_decode_keras():
    print("TBD")

def beam_search_decode():
    print("TBD")

def beam_search_decode_with_model():
    print("TBD")

def compute_mean_edit_distance(model, dataset, verbose=False):
    distances = []
    for batch in dataset:
        inputs = batch[0]["inputs"]
        labels = batch[0]["labels"]
        input_lengths = batch[0]["input_length"]
        label_lengths = batch[0]["label_length"]

        # Pass test data into network
        softmax_out = model.predict(inputs)

        # CTC decoding of network outputs
        prediction = K.ctc_decode(softmax_out, input_lengths, greedy=True, beam_width=100, top_paths=1)
        prediction = K.get_value(prediction[0][0])

        for i, pred_label in enumerate(prediction):
            label = labels[i]
            label_length = label_lengths[i]
            label = _to_int_list(label)
            label = _label_to_sequence(label, label_length)

            pred_label = _to_int_list(pred_label)
            pred_label_len = _calculate_len_pred(pred_label)
            pred_label = _label_to_sequence(pred_label, pred_label_len)

            edit_dist = levenshtein.normalized_distance(label, pred_label)
            distances.append(edit_dist)

            if verbose == True:
                print("True label: {0}".format(label))
                print("Predicted label: {0}".format(pred_label))
                print("Edit distance: {0}\n\n\n".format(edit_dist))
    
    return mean(distances)

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