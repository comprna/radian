import numpy as np

from tensorflow.io.gfile import glob
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from data import get_dataset

def main():
    config = get_config('config.yaml')

    # Get test data
    test_files = glob("{0}/test/*.tfrecords".format(shards_dir))
    test_dataset = get_dataset(test_files, config, val=False) # Modify get_dataset for test data??

    # Load finalized model
    saved_filepath = './saved_model'
    model = load_model(saved_filepath, compile=True)

    # Pass test data into network
    softmax_out = model.predict(test_dataset)

    # CTC decoding of network outputs
    prediction = K.ctc_decode(softmax_out, input_length, greedy=True, beam_width=100, top_paths=1)
    prediction = K.get_value(prediction[0][0])[0]
    print(prediction)

    # [OPTIONAL] Assemble into reads


    # Compute error rate
    # Use Levenshtein (or Hamming?) normalised similarity here: https://pypi.org/project/textdistance/


    # Refs:
    # https://www.programcreek.com/python/example/122027/keras.backend.ctc_decode
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
    # https://machinelearningmastery.com/train-final-machine-learning-model/
    # https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/

    print("Test")

if __name__ == "__main__":
    main()
