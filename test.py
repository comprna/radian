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

        for i, p in enumerate(prediction):
            signal = inputs[i]
            label = labels[i]
            label_length = label_lengths[i]
            y_pred = softmax_out[i]
            print("Label: {0}".format(label))
            print("Label length: {0}".format(label_length))
            label = _label_to_sequence(label, label_length)
            print("Label after format: {0}".format(label))

            print("Predicted sequence: {0}".format(p))
            p = _label_to_sequence(p, label_length)
            print("Predicted sequence after format: {0}".format(p))

            print("Softmax output: {0}".format(y_pred))
            print("\n\n\n")
            edit_dist = levenshtein.normalized_distance(label, p)
            print("Edit distance: {0}".format(edit_dist))
            break

def _label_to_sequence(label, label_length):
    print(label)
    label = label[:label_length]
    print(label)
    bases = ['A', 'C', 'G', 'T']
    label = list(map(lambda b: bases[b], label))
    print(label)
    return "".join(label)



    # [OPTIONAL] Assemble into reads


    # Compute error rate
    # Use Levenshtein (or Hamming?) normalised similarity here: https://pypi.org/project/textdistance/


    # Refs:
    # https://www.programcreek.com/python/example/122027/keras.backend.ctc_decode
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
    # https://machinelearningmastery.com/train-final-machine-learning-model/
    # https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/

if __name__ == "__main__":
    main()
