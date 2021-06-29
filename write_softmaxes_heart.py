import numpy as np
from tensorflow.keras import backend as K

from model import get_prediction_model
from utilities import get_config, setup_local

def main():
    setup_local()
    s_config_file = '/g/data/xc17/Eyras/alex/working/global_decoding/train-3-37/data/all_val/6_WriteSoftmaxes/copied_files/s-config-3.yaml'
    s_model_file = '/g/data/xc17/Eyras/alex/working/global_decoding/train-3-37/data/all_val/6_WriteSoftmaxes/copied_files/train-3-model-10.h5'

    s_config = get_config(s_config_file)
    s_model = get_prediction_model(s_model_file, s_config)

    # Read val data from numpy array
    data_dir = "/g/data/xc17/Eyras/alex/working/global_decoding/train-3-37/data/all_val/6_WriteSoftmaxes/copied_files"
    with open(f"{data_dir}/heart_read_windows_all.npy", "rb") as f:
        windows_per_read = np.load(f, allow_pickle=True)

    print(f"Number of reads: {len(windows_per_read)}")
    i = 0
    softmaxes_per_read = []
    for read in windows_per_read:
        softmaxes = []
        for window in read:
            softmax = s_model.predict(np.array([window,]))
            softmax = np.squeeze(softmax)
            softmaxes.append(softmax)
        print(i)
        i += 1
        softmaxes_per_read.append(softmaxes)

    # Write to file
    with open("softmaxes_per_read_heart.npy", "wb") as f:
        np.save(f, softmaxes_per_read)


if __name__ == "__main__":
    main()
