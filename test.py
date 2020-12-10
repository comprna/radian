from tensorflow.io.gfile import glob

from data import get_dataset
from edit_distance import compute_mean_edit_distance
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    # setup_local()
    config = get_config('/home/150/as2781/rnabasecaller/config.yaml')

    test_files = glob("/g/data/xc17/Eyras/alex/rna-basecaller/shards/debugging/mixed-labels-10/val/*.tfrecords")
    test_dataset = get_dataset(test_files, config, val=True)

    saved_filepath = '/g/data/xc17/Eyras/alex/rna-basecaller/train-42/model-152.h5'
    model = get_prediction_model(saved_filepath, config)

    # TODO: Assemble into reads

    mean_ed = compute_mean_edit_distance(model, test_dataset, verbose=True)
    print("Mean edit distance on test data: {0}".format(mean_ed))

if __name__ == "__main__":
    main()
