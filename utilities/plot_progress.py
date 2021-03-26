import glob
import os

from tensorflow.io import gfile

from data import get_dataset
from evaluate import predict_greedy
from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()
    config = get_config('/home/alex/Documents/rnabasecaller/config.yaml')

    data_files = gfile.glob("/mnt/sda/singleton-dataset-generation/dRNA/4_8_NNInputs/0_2_CreateTFRecords/2_WriteTFRecords/shards/val/*.tfrecords")
    dataset = get_dataset(data_files, config.train.batch_size, val=True)

    model_folder = "/mnt/sda/rna-basecaller/experiments/4_8_NNInputs/train-1"
    model_files = glob.glob(os.path.join(model_folder, "*.h5"))
    for model_file in sorted(model_files):
        model_id = model_file.split("/")[-1].split(".")[0]
        model = get_prediction_model(model_file, config)
        predict_greedy(model, dataset, plot=True, model_id=model_id)

# Command to view evolution of spikes for a single training instance
# eog model-*-2.png (This would open all plots for instance 2)

if __name__ == "__main__":
    main()
