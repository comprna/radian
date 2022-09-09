import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

def assemble_matrices(matrices, step_size):
    # Vertically stack the overlapping matrices together
    vstack = create_vstack(matrices, step_size)
    # Collapse the stack into a single matrix for the entire read
    return collapse_vstack(vstack)

def create_vstack(read_matrices, step_size):
    # Vertically stack matrices by their position in the original read
    # The result is a list of matrices per timestep in the original read
    vstack = []
    t_start = 0
    for batch_matrices in read_matrices:
        for matrix in batch_matrices:
            # Start at the appropriate timestep
            t_curr = t_start

            for dist in matrix:
                # Extend stack with current timestep
                if t_curr >= len(vstack):
                    vstack.append([])
                
                # Add distribution to current timestep in stack
                vstack[t_curr].append(dist)

                # Increment current time
                t_curr += 1

            # Once all distributions added, increment t_start by step_size
            t_start += step_size
    return vstack

def collapse_vstack(vstack):
    global_matrix = []
    for dist_list in vstack:         
        # Combine all distributions at the current timestep
        if len(dist_list) > 1:
            global_matrix.append(average_dist(dist_list))
        else:
            global_matrix.append(dist_list[0])
    return np.asarray(global_matrix)

def average_dist(dists):
    result = dists[0]
    for i, dist in enumerate(dists):
        # Already added first item before loop
        if i == 0:
            continue
        np.add(result, dist)
    return normalize([result], norm="l1")[0]

def plot_assembly(matrices, global_matrix, window_size, step_size):
    display_windows = 5 # Only show a few windows for ease of display
    matrices = matrices[0][:display_windows]
    len_global = window_size + (display_windows - 1) * step_size
    global_matrix = global_matrix[:len_global]
    pad_size_before = 0
    pad_size_after = len_global - window_size
    padded_windows = []
    for matrix in matrices:
        padded_window = np.pad(matrix, ((pad_size_before, pad_size_after), (0,0)), "constant")
        padded_windows.append(padded_window)
        pad_size_before += step_size
        pad_size_after -= step_size
    plot(padded_windows, global_matrix, display_windows)

def plot(padded_windows, global_matrix, display_windows):
    _, axs = plt.subplots(display_windows+1, 1, sharex="all")
    for i, matrix in enumerate(padded_windows):
        if i >= display_windows:
            break
        axs[i].imshow(np.transpose(matrix), cmap="gray_r", aspect="auto")
    axs[-1].imshow(np.transpose(global_matrix), cmap="gray_r", aspect="auto")
    plt.show()