import numpy as np
from beam_search_decoder import ctcBeamSearch

def main():
    # Read test data from file

    data_dir = "/g/data/xc17/Eyras/alex/working/global_decoding/train-3-37/data/all_val/7_TestLocalDecoding/1_DecodeWindowSoftmaxes"
    softmaxes_file = f"{data_dir}/copied_files/softmaxes_per_read_hek293.npy"
    with open(softmaxes_file, "rb") as f:
        read_softmaxes_all = np.load(f, allow_pickle=True)

    # Decode softmaxes with beam search

    classes = 'ACGT'
    rna_model = None
    read_preds_all = []
    i = 1
    for read in read_softmaxes_all:
        # Every 100 reads write predictions to file
        if i % 100 == 0:
            with open(f"{data_dir}/hek293_read_preds_all_{i}.npy", "wb") as f:
                np.save(f, read_preds_all)
            read_preds_all = []

        print(len(read))
        preds = []
        j = 1
        for softmax in read:
            pred = ctcBeamSearch(softmax, classes, rna_model, None)
            preds.append(pred)
            print(j)
            j += 1
        read_preds_all.append(preds)
        i += 1
    
    # Save any remaining decoded reads
    with open(f"{data_dir}/hek293_read_preds_all_{i}.npy", "wb") as f:
        np.save(f, read_preds_all)


if __name__ == "__main__":
    main()