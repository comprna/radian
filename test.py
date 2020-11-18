import matplotlib.pyplot as plt
import numpy as np

from tensorflow.io.gfile import glob
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from data import get_dataset
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()

    config = get_config('/home/150/as2781/rnabasecaller/config.yaml')
    # shards_dir = '/home/alex/OneDrive/phd-project/singleton-dataset-generation/dRNA/3_8_NNInputs/debugging/single-label/CATTTTATCTCTGGGTCATT/1000-instances'
    shards_dir = '/g/data/xc17/Eyras/alex/rna-basecaller/shards/debugging/single-label/CATTTTATCTCTGGGTCATT/1000-instances'

    # Get test data
    test_files = glob("{0}/*.tfrecords".format(shards_dir))
    test_dataset = get_dataset(test_files, config, val=True)

    # Load finalized model
    # saved_filepath = '/home/alex/OneDrive/phd-project/rna-basecaller/train-7-local/model-500.h5'
    saved_filepath = '/g/data/xc17/Eyras/alex/rna-basecaller/train-8/model-1000.h5'
    model = get_prediction_model(saved_filepath, config)

    i = 1
    for batch in test_dataset:
        if i > 10:
            break # Just go through 10 batches
        inputs = batch[0]["inputs"]
        labels = batch[0]["labels"]
        input_length = batch[0]["input_length"]
        print(inputs)
        print(inputs.shape) # (256, 512)

        # Pass test data into network
        softmax_out = model.predict(inputs)
        print(softmax_out.shape) # (256, 512, 5)

        # CTC decoding of network outputs
        prediction = K.ctc_decode(softmax_out, input_length, greedy=False, beam_width=100, top_paths=1)
        print(prediction)
        prediction = K.get_value(prediction[0][0])
        print(prediction)

        for i, p in enumerate(prediction):
            signal = inputs[i]
            label = labels[i]
            y_pred = softmax_out[i]
            # plt.plot(signal)
            # plt.show()
            print(p)
            print(y_pred)
        
        i += 1

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
