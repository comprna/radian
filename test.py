from tensorflow.io.gfile import glob

from data import get_dataset
from edit_distance import compute_mean_edit_distance_greedy
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()
    config = get_config('/home/alex/Documents/rnabasecaller/config.yaml')

    test_files = glob("/mnt/sda/singleton-dataset-generation/dRNA/4_8_NNInputs/0_2_CreateTFRecords/2_WriteTFRecords/shards/val/*.tfrecords")
    test_dataset = get_dataset(test_files, config.train.batch_size, val=True)

    saved_filepath = '/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/train-6/model-01.h5'
    model = get_prediction_model(saved_filepath, config)

    # TODO: Assemble into reads

    mean_ed = compute_mean_edit_distance_greedy(model, test_dataset, verbose=True)
    print("Mean edit distance on test data: {0}".format(mean_ed))

if __name__ == "__main__":
    main()
