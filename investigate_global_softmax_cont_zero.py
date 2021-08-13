import numpy as np

def main():
    data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test"
    with open(f"{data_dir}/6_WriteSoftmaxes/softmaxes_per_read_heart.npy", "rb") as f:
        read_softmaxes = np.load(f, allow_pickle=True)


    for softmax in read_softmaxes[4]:
        for row in softmax:
            for pr in row:
                if pr == 0:
                    print("Local softmax contains a 0!")


if __name__ == "__main__":
    main()