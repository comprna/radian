import os
from pathlib import Path

import numpy as np

from sequence_assembly import simple_assembly, index2base

def main():
    # Load local window predictions

    data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test/7_TestLocalDecoding/1_DecodeWindowSoftmaxes/heart/numpy"

    # Combine all reads together so we can iterate more easily

    npy_paths = sorted(Path(data_dir).iterdir(), key=os.path.getctime) # ctime is last change to metadata (e.g. filename)
    with open(npy_paths[0], "rb") as f:
        read_local_preds = np.load(f, allow_pickle=True)
    for path in npy_paths[1:]:
        with open(path, "rb") as f:
            read_local_preds = np.concatenate((read_local_preds, np.load(f, allow_pickle=True)))
    print(len(read_local_preds))

    # Assemble window predictions into consensus prediction

    read_assembled_preds = []
    for preds in read_local_preds:
        consensus = simple_assembly(preds)
        consensus_seq = index2base(np.argmax(consensus, axis=0))
        read_assembled_preds.append(consensus_seq)
    with open(f"{data_dir}/read_assembled_preds_heart.npy", "wb") as f:
        np.save(f, read_assembled_preds)

if __name__ == "__main__":
    main()