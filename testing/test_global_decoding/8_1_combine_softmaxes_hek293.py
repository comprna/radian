import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

STEP_SIZE = 128
WINDOW_LEN = 1024
OVERLAP = WINDOW_LEN-STEP_SIZE
N_BASES = 4
N_WINDOWS = 30

def visualise_assembly(softmax_windows, global_softmax):
    softmax_windows = softmax_windows[:N_WINDOWS]
    len_global = WINDOW_LEN + (N_WINDOWS - 1) * STEP_SIZE
    global_softmax = global_softmax[:len_global]
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

def sum_normalised_list_l1(dists):
    result = dists[0]
    for i, dist in enumerate(dists):
        # Already added first item before loop
        if i == 0:
            continue
        np.add(result, dist)
    result = normalize([result], norm="l1")[0]
    return result

def main():
    # Read test data from file

    # data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/data/all_val"
    # with open(f"{data_dir}/6_WriteSoftmaxes/softmaxes_per_read_hek293.npy", "rb") as f:
    #     read_softmaxes = np.load(f, allow_pickle=True)

    # data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/PoC_test/data"
    # with open(f"{data_dir}/softmaxes_all.npy", "rb") as f:
    #     read_softmaxes = np.load(f, allow_pickle=True)

    data_dir = "/mnt/sda/rna-basecaller/experiments/global-decoding/train-3-37/val_test"
    with open(f"{data_dir}/6_WriteSoftmaxes/softmaxes_per_read_hek293.npy", "rb") as f:
        read_softmaxes = np.load(f, allow_pickle=True)

    # Stack overlapping softmaxes to reconstruct expanded version of
    # global softmax (so that there are multiple distributions per
    # timestep)

    read_global_softmaxes = []
    i = 1
    for read in read_softmaxes:

        # Every 100 reads write global softmaxes to file

        if i % 100 == 0:
            with open(f"{data_dir}/hek293_read_global_softmaxes_{i}.npy", "wb") as f:
                np.save(f, read_global_softmaxes)
            read_global_softmaxes = []

        # Stack all softmaxes together (jagged array - a list per timestep)
        
        stack = []
        t_start = 0
        # TODO: Is list length a method or attribute of list??? If method, then store variable here
        for softmax in read:
            # Start at the appropriate timestep
            t_curr = t_start

            for dist in softmax:
                # Check if current timestep is already in the list
                if t_curr >= len(stack):
                    # Add timestep if it isn't already in the list
                    stack.append([])

                # Add distribution to current timestep
                stack[t_curr].append(dist)

                # Increment current time
                t_curr += 1

            # Once all distributions added, increment t_start by 128
            t_start += 128
        
        # Now that we have the stack, collapse it to get the final
        # global softmax so that there is only one distribution per
        # timestep.

        global_softmax = []
        for t, dist_list in enumerate(stack):         
            # Combine all distributions at the current timestep
            # Approach 2:
            if len(dist_list) > 1:
                global_softmax.append(sum_normalised_list_l1(dist_list))
            else:
                global_softmax.append(dist_list[0])
        global_softmax = np.asarray(global_softmax)
        read_global_softmaxes.append(global_softmax)

        i += 1

        # visualise_assembly(read, global_softmax)

    # Write remaining global softmaxes to file

    with open(f"{data_dir}/hek293_read_global_softmaxes_{i}.npy", "wb") as f:
        np.save(f, read_global_softmaxes)


if __name__ == "__main__":
    main()