from tensorflow.io.gfile import glob

from data import get_dataset
from edit_distance import compute_mean_edit_distance
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()
    config = get_config('/home/alex/Documents/rnabasecaller/config.yaml')

    test_files = glob("/mnt/sda/singleton-dataset-generation/dRNA/3_8_NNInputs/tfrecord_approach/shards/debugging/mixed-labels-5/val/*.tfrecords")
    test_dataset = get_dataset(test_files, config.train.batch_size, val=True)

    saved_filepath = '/home/alex/OneDrive/phd-project/rna-basecaller/train-53/model-60.h5'
    model = get_prediction_model(saved_filepath, config)

    # TODO: Assemble into reads

    mean_ed = compute_mean_edit_distance(model, test_dataset, verbose=True)
    print("Mean edit distance on test data: {0}".format(mean_ed))

if __name__ == "__main__":
    main()
