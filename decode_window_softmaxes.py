import json

def main():
    # Read test data from file

    data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/data/all_val"
    # with open(f"{data_dir}/4_Construct_GT_Label_Per_Read/hek293_read_gt_labels.json", "r") as f:
    #     read_gt_labels = json.load(f)
    # with open(f"{data_dir}/5_WriteToNumpy/hek293/hek293_read_ids.npy", "rb") as f:
    #     read_ids = np.load(f)
    # with open(f"{data_dir}/5_WriteToNumpy/hek293/hek293_read_windows_all.npy", "rb") as f:
    #     read_windows_all = np.load(f, allow_pickle=True)
    # with open(f"{data_dir}/5_WriteToNumpy/hek293/hek293_read_labels_all.npy", "rb") as f:
    #     read_labels_all = np.load(f, allow_pickle=True)
    with open(f"{data_dir}/6_WriteSoftmaxes/softmaxes_per_read_hek293.npy", "rb") as f:
        read_softmaxes_all = np.load(f, allow_pickle=True)


    ######################## APPROACH 1 ###############################

    # Decode softmaxes with beam search

    classes = 'ACGT'
    rna_model = None
    read_preds_all = []
    for read in read_softmaxes_all:
        print(len(read))
        preds = []
        j = 1
        for softmax in read:
            pred, _ = ctcBeamSearch(softmax, classes, rna_model, None)
            preds.append(pred)
            print(j)
            j += 1
        read_preds_all.append(preds)
    with open(f"{data_dir}/read_preds_all.npy", "wb") as f:
        np.save(f, read_preds_all)
    # with open(f"{data_dir}/7_TestDecoding/read_preds_all.npy", "rb") as f:
    #     read_preds_all = np.load(f)


if __name__ == "__main__":
    main()