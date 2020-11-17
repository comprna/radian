import numpy as np

from tensorflow.io.gfile import glob
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from data import get_dataset
from model import load_checkpoint
from utilities import get_config, setup_local

def main():
    setup_local()

    config = get_config('config.yaml')
    shards_dir = '/mnt/sda/singleton-dataset-generation/dRNA/3_8_NNInputs/tfrecord_approach/shards'

    # Get test data
    test_files = glob("{0}/debug/*.tfrecords".format(shards_dir))
    test_dataset = get_dataset(test_files, config, val=True)

    # Load finalized model
    saved_filepath = '/home/alex/OneDrive/phd-project/rna-basecaller/train-4/model-04-27.29.h5'
    model = load_checkpoint(saved_filepath)

    softmax_out = model.predict(test_dataset, steps=1)
    print(softmax_out.shape)

    # Pass test data into network
    for sample in test_dataset:
        inputs = sample[0]['inputs']
        labels = sample[0]['labels']
        input_lengths = sample[0]['input_length']
        for i, _ in enumerate(inputs):
            signal = inputs[i]
            signal_length = input_lengths[i]
            label = labels[i]
            print(signal)
            print(label)


            # CTC decoding of network outputs
            prediction = K.ctc_decode(softmax_out, signal_length, greedy=True, beam_width=100, top_paths=1)
            print(prediction)
            prediction = K.get_value(prediction[0][0])
            print(prediction)

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
