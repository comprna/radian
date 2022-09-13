import numpy as np


def get_windows(signal, window_size, step_size):
    if step_size <= 0:
        raise ValueError("Step size must be > 0")
    if step_size > window_size:
        raise ValueError("Step size must be <= window size")
    windows = []
    start = 0
    while start + window_size <= signal.shape[0]:
        window = signal[start:start + window_size]
        windows.append(window)
        start += step_size

    # Pad last window
    last = signal[start:]
    pad_end = window_size - len(last)
    last = np.pad(last, (0, pad_end))
    windows.append(last)

    return np.asarray(windows), pad_end

def mad_normalise(signal, outlier_z_score):
    if signal.shape[0] == 0:
        raise ValueError("Signal must not be empty to normalise")
    median = np.median(signal)
    mad = _calculate_mad(signal, median)
    vnormalise_value = np.vectorize(_normalise_value)
    return vnormalise_value(np.array(signal), median, mad, outlier_z_score)

def _calculate_mad(signal, median):
    f = lambda x, median: np.abs(x - median)
    distances_from_median = f(signal, median)
    return np.median(distances_from_median)

def _normalise_value(x, median, mad, outlier_z_score):
    modified_z_score = _calculate_modified_z_score(x, median, mad)
    if modified_z_score > outlier_z_score:
        return outlier_z_score
    elif modified_z_score < -1 * outlier_z_score:
        return -1 * outlier_z_score
    else:
        return modified_z_score

def _calculate_modified_z_score(x, median, mad):
    return (x - median) / (1.4826 * mad)