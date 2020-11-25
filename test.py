from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.io.gfile import glob
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from textdistance import levenshtein

from data import get_dataset
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()
    config = get_config('/home/150/as2781/rnabasecaller/config.yaml')

    # Get test data
    test_files = glob("/g/data/xc17/Eyras/alex/rna-basecaller/shards/debugging/CATTTTATCTCTGGGTCATT_GCCTACTTCGTCTATCACTCCT/*.tfrecords")
    test_dataset = get_dataset(test_files, config, val=True)

    # Load finalized model
    saved_filepath = '/g/data/xc17/Eyras/alex/rna-basecaller/train-16/model-389.h5'
    model = get_prediction_model(saved_filepath, config)

    # TODO: Assemble into reads

    distances = []

    for batch in test_dataset:
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
            signal = inputs[i]
            label = labels[i]
            label_length = label_lengths[i]
            y_pred = softmax_out[i]

            label = _to_int_list(label)
            label = _label_to_sequence(label, label_length)
            print("True label: {0}".format(label))

            pred_label = _to_int_list(pred_label)
            pred_label_len = _calculate_len_pred(pred_label)
            pred_label = _label_to_sequence(pred_label, pred_label_len)
            print("Predicted label: {0}".format(pred_label))

            edit_dist = levenshtein.normalized_distance(label, pred_label)
            print("Edit distance: {0}".format(edit_dist))
            distances.append(edit_dist)
            print("\n\n\n")
            break
    
    print(distances)
    print(mean(distances))

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
    main()

    # Refs:
    # https://www.programcreek.com/python/example/122027/keras.backend.ctc_decode
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
    # https://machinelearningmastery.com/train-final-machine-learning-model/
    # https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/
