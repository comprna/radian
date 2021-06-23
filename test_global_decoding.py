import numpy as np
from statistics import mean
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt
from scipy.stats import entropy
from textdistance import levenshtein

from beam_search_decoder import ctcBeamSearch
from easy_assembler import simple_assembly, index2base

STEP_SIZE = 128
WINDOW_LEN = 1024
OVERLAP = WINDOW_LEN-STEP_SIZE
N_BASES = 4
N_WINDOWS = 30

def max_entropy(dist_a, dist_b):
    entropy_a = entropy(dist_a)
    entropy_b = entropy(dist_b)

    # Take the column with lowest entropy (least surprise, i.e. 
    # most confident prediction).
    if entropy_b > entropy_a:
        return dist_b
    else:
        return dist_a

def conflate(dist_a, dist_b):
    num = np.multiply(dist_a, dist_b)
    den = np.sum(num)
    return np.divide(num, den)

def conflate_list(dists):
    num = dists[0]
    for i, dist in enumerate(dists):
        if i == 0:
            continue
        num = np.multiply(num, dist)
    den = np.sum(num)
    return np.divide(num, den)

def sum_normalised_list_l2(dists):
    result = dists[0]
    for i, dist in enumerate(dists):
        # Already added first item before loop
        if i == 0:
            continue
        np.add(result, dist)
    result = normalize([result], norm="l2")[0]
    return result

def sum_normalised_list_l1(dists):
    result = dists[0]
    for i, dist in enumerate(dists):
        # Already added first item before loop
        if i == 0:
            continue
        np.add(result, dist)
    result = normalize([result], norm="l1")[0]
    return result

def sum_normalised_l2(dist_a, dist_b):
    result = np.add(dist_a, dist_b)
    result = normalize([result], norm="l2")[0]
    return result

def sum_normalised_l1(dist_a, dist_b):
    result = np.add(dist_a, dist_b)
    result = normalize([result], norm="l1")[0]
    return result

def min_entropy(dist_a, dist_b):
    entropy_a = entropy(dist_a)
    entropy_b = entropy(dist_b)

    # Take the column with lowest entropy (least surprise, i.e. 
    # most confident prediction).
    if entropy_b < entropy_a:
        return dist_b
    else:
        return dist_a

def combine(global_softmax, new_softmax):
    # Last OVERLAP timesteps in global_softmax overlaps with first
    # OVERLAP timesteps in new_softmax.  The overlapping region needs
    # to be combined.
    last = global_softmax[-1*OVERLAP:,:]
    first = new_softmax[:OVERLAP,:]

    # fig, axs = plt.subplots(3, 1)
    # axs[0].imshow(np.transpose(global_softmax), cmap="gray_r", aspect="auto")
    # axs[1].imshow(np.transpose(last), cmap="gray_r", aspect="auto")
    # axs[2].imshow(np.transpose(first), cmap="gray_r", aspect="auto")
    # plt.show()

    # fig, axs = plt.subplots(2, 1, sharex="all")
    # axs[0].plot(np.transpose(last)[4], label="b", color="grey", linestyle="dashed")
    # axs[0].plot(np.transpose(last)[0], label="A", color="red")
    # axs[0].plot(np.transpose(last)[1], label="C", color="orange")
    # axs[0].plot(np.transpose(last)[2], label="G", color="green")
    # axs[0].plot(np.transpose(last)[3], label="T", color="blue")
    # axs[0].legend()
    # axs[1].plot(np.transpose(first)[4], label="b", color="grey", linestyle="dashed")
    # axs[1].plot(np.transpose(first)[0], label="A", color="red")
    # axs[1].plot(np.transpose(first)[1], label="C", color="orange")
    # axs[1].plot(np.transpose(first)[2], label="G", color="green")
    # axs[1].plot(np.transpose(first)[3], label="T", color="blue")
    # axs[1].legend()
    # plt.show()

    # Combine the overlapping regions
    combined = np.zeros((OVERLAP, N_BASES+1))
    for t, _ in enumerate(last):
        # combined[t] = min_entropy(last[t], first[t])
        # combined[t] = sum_normalised_l1(last[t], first[t])
        # combined[t] = sum_normalised_l2(last[t], first[t])
        # combined[t] = conflate(last[t], first[t])
        combined[t] = max_entropy(last[t], first[t])

    fig, axs = plt.subplots(3, 1, sharex="all")
    axs[0].imshow(np.transpose(last), cmap="gray_r", aspect="auto")
    axs[1].imshow(np.transpose(first), cmap="gray_r", aspect="auto")
    axs[2].imshow(np.transpose(combined), cmap="gray_r", aspect="auto")
    plt.show()

    # Update the global softmax with the overlap results
    global_softmax[-1*OVERLAP:] = combined

    # Append the non-overlapping portion of the new softmax to global
    global_softmax = np.concatenate((global_softmax, new_softmax[-1*STEP_SIZE:,:]))

    return global_softmax

def visualise_assembly(softmax_windows, global_softmax):
    len_global = WINDOW_LEN + (N_WINDOWS - 1) * STEP_SIZE
    pad_size_before = 0
    pad_size_after = len_global - WINDOW_LEN
    padded_windows = []
    for softmax in softmax_windows:
        padded_window = np.pad(softmax, ((pad_size_before, pad_size_after), (0,0)), "constant")
        padded_windows.append(padded_window)
        pad_size_before += STEP_SIZE
        pad_size_after -= STEP_SIZE
    plot_assembly(padded_windows, global_softmax)

def plot_assembly(padded_windows, global_softmax):
    # Only show first n windows for ease of viewing
    n_windows_to_view = 5
    _, axs = plt.subplots(n_windows_to_view+1, 1, sharex="all")
    for i, softmax in enumerate(padded_windows):
        if i >= n_windows_to_view:
            break
        axs[i].imshow(np.transpose(softmax), cmap="gray_r", aspect="auto")
    axs[-1].imshow(np.transpose(global_softmax), cmap="gray_r", aspect="auto")
    plt.show()

def main():
    # Read test data from file

    # data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/data"
    data_dir = "decoding_test"
    with open(f"{data_dir}/orig_signals.npy", "rb") as f:
        orig_signals = np.load(f)
    with open(f"{data_dir}/gt_labels.npy", "rb") as f:
        gt_labels = np.load(f)
    with open(f"{data_dir}/windows_all.npy", "rb") as f:
        windows_all = np.load(f)
    with open(f"{data_dir}/window_labels_all.npy", "rb") as f:
        window_labels_all = np.load(f)
    with open(f"{data_dir}/softmaxes_all.npy", "rb") as f:
        softmaxes_all = np.load(f)


    ######################## APPROACH 1 ###############################

    # Decode softmaxes with beam search

    # classes = 'ACGT'
    # rna_model = None
    # preds_all = []
    # for softmaxes in softmaxes_all:
    #     preds = []
    #     for softmax in softmaxes:
    #         pred, _ = ctcBeamSearch(softmax, classes, rna_model, None)
    #         preds.append(pred)
    #     preds_all.append(preds)
    # with open(f"{data_dir}/preds_all.npy", "wb") as f:
    #     np.save(f, preds_all)
    with open(f"{data_dir}/preds_all.npy", "rb") as f:
        preds_all = np.load(f)

    # Assemble window predictions

    # preds_windows = []
    # for i, preds in enumerate(preds_all):
    #     consensus = simple_assembly(preds)
    #     consensus_seq = index2base(np.argmax(consensus, axis=0))
    #     preds_windows.append(consensus_seq)
    # with open(f"{data_dir}/preds_windows.npy", "wb") as f:
    #     np.save(f, preds_windows)
    with open(f"{data_dir}/preds_windows.npy", "rb") as f:
        preds_windows = np.load(f)

    ######################## APPROACH 2 ###############################

    # Stack overlapping softmaxes to reconstruct expanded version of
    # global softmax (so that there are multiple distributions per
    # timestep)

    all_global_expanded = []
    for softmaxes in softmaxes_all:
        # Stack all softmaxes together (jagged array - a list per timestep)
        global_expanded = []
        t_start = 0
        # TODO: Is list length a method or attribute of list??? If method, then store variable here
        for softmax in softmaxes:
            # Start at the appropriate timestep
            t_curr = t_start

            for dist in softmax:
                # Check if current timestep is already in the list
                if t_curr >= len(global_expanded):
                    # Add timestep if it isn't already in the list
                    global_expanded.append([])

                # Add distribution to current timestep
                global_expanded[t_curr].append(dist)

                # Increment current time
                t_curr += 1

            # Once all distributions added, increment t_start by 128
            t_start += 128

        all_global_expanded.append(global_expanded)

    # Collapse stack of softmaxes to get final global softmax so that 
    # there is only one distribution per timestep

    all_global_collapsed = []
    for global_expanded in all_global_expanded:
        global_collapsed = []
        for t, dist_list in enumerate(global_expanded):         
            # Combine all distributions at the current timestep
            # Approach 2:
            if len(dist_list) > 1:
                global_collapsed.append(sum_normalised_list_l2(dist_list))
                # global_collapsed.append(sum_normalised_list_l1(dist_list))
                # global_collapsed.append(conflate_list(dist_list))
            # Approach 3: Exclude last dist, which has most uncertainty
            # if len(dist_list) > 2:
            #     # global_collapsed.append(sum_normalised_list_l2(dist_list[:-1]))
            #     global_collapsed.append(sum_normalised_list_l1(dist_list[:-1]))
            #     # global_collapsed.append(conflate_list(dist_list[:-1]))
            else:
                global_collapsed.append(dist_list[0])
        global_collapsed = np.asarray(global_collapsed)
        all_global_collapsed.append(global_collapsed)

    # Visualise softmax assembly
    for i, _ in enumerate(softmaxes_all):
        visualise_assembly(softmaxes_all[i], all_global_collapsed[i])

    # Decode global matrix with beam search

    classes = 'ACGT'
    rna_model = None
    preds_global = []
    for softmax in all_global_collapsed:
        pred, _ = ctcBeamSearch(softmax, classes, rna_model, None)
        preds_global.append(pred)
    # with open(f"{data_dir}/preds_global_min_entropy.npy", "wb") as f:
    #     np.save(f, preds_global)
    # with open(f"{data_dir}/preds_global_min_entropy.npy", "rb") as f:
    #     preds_global = np.load(f)

    ########################## ANALYSIS ###############################

    # Compute edit distances

    eds_windows = []
    eds_global = []
    for i, _ in enumerate(preds_windows):
        print(f"Ground truth: {gt_labels[i]}")
        print(f"Window pred:  {preds_windows[i]}")
        print(f"Global pred:  {preds_global[i]}")
        ed_windows = levenshtein.normalized_distance(gt_labels[i], preds_windows[i])
        ed_global = levenshtein.normalized_distance(gt_labels[i], preds_global[i])
        eds_windows.append(ed_windows)
        eds_global.append(ed_global)
        print(f"Window ED: {ed_windows}")
        print(f"Global ED: {ed_global}\n\n")

    print(f"Mean window approach: {mean(eds_windows)}")
    print(f"Mean global approach: {mean(eds_global)}")

if __name__ == "__main__":
    main()