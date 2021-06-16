import numpy as np

from beam_search_decoder import ctcBeamSearch


def main():
    # Read test data from file
    with open("orig_signals.npy", "rb") as f:
        orig_signals = np.load(f)
    with open("gt_labels.npy", "rb") as f:
        gt_labels = np.load(f)
    with open("windows_all.npy", "rb") as f:
        windows_all = np.load(f)
    with open("window_labels_all.npy", "rb") as f:
        window_labels_all = np.load(f)
    with open("softmaxes_all.npy", "rb") as f:
        softmaxes_all = np.load(f)


    # APPROACH 1

    # Decode softmaxes with beam search

    classes = 'ACGT'
    rna_model = None
    preds_all = []
    for softmaxes in softmaxes_all:
        preds = []
        for softmax in softmaxes:
            pred, _ = ctcBeamSearch(softmax, classes, rna_model, None)
            preds.append(pred)
        preds_all.append(preds)
    with open("preds_all.npy", "wb") as f:
        np.save(f, preds_all)

    # Assemble window predictions



    # APPROACH 2

    # Assemble softmaxes

    
    # Decode global matrix with beam search


if __name__ == "__main__":
    main()