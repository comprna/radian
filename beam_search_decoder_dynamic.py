"""
This has been adapted from https://github.com/githubharald/CTCDecoder/blob/master/ctc_decoder/beam_search.py
"""

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np
import tensorflow as tf


N_BASES = 4

def log(x: float) -> float:
    return -math.inf if x == 0 else math.log(x)


@dataclass
class BeamEntry:
    """Information about one single beam at specific time-step."""
    pr_total: float = log(0)  # blank and non-blank
    pr_non_blank: float = log(0)  # non-blank
    pr_blank: float = log(0)  # blank
    labeling: tuple = ()  # beam-labeling
    indices: tuple = ()  # indices of each character in beam


class BeamList:
    """Information about all beams at specific time-step."""

    def __init__(self) -> None:
        self.entries = defaultdict(BeamEntry)

    def sort_labelings(self) -> List[Tuple[int]]:
        """Return beam-labelings, sorted by probability."""
        beams = self.entries.values()
        sorted_beams = sorted(beams, reverse=True, key=lambda x: x.pr_total)
        return [x.labeling for x in sorted_beams], [x.indices for x in sorted_beams]


def get_context(labeling, len_context, exclude_last=False):
    # the context is the last portion of the beam
    if exclude_last == True:
        context = labeling[-(len_context+1):-1]
    else:
        context = labeling[-len_context:]

    return context


def get_next_base_prob(context, model, cache):
    if context not in cache:
        # convert context into RNA model input format
        context_arr = tf.one_hot(list(context), N_BASES).numpy()
        context_arr = context_arr.reshape(1, len(context), N_BASES)

        # predict the distribution of the next base given the context
        dist = model.predict(context_arr)[0]
        cache[context] = dist
    else:
        dist = cache[context]
    
    return dist


def combine_dists(r_dist, s_dist):
    # get the base (i.e. non-blank) distribution from the signal model
    s_base_prob = np.sum(s_dist[:-1])
    s_base_dist = s_dist[:-1] / s_base_prob

    # average the signal and rna model probs
    c_dist = np.add(r_dist, s_base_dist) / 2

    # reconstruct the signal model distribution (including blank)
    c_dist = c_dist * s_base_prob
    c_dist = np.append(c_dist, s_dist[-1])

    return c_dist


def normalise(dist):
    if sum(dist) == 0:
        return dist
    return dist / sum(dist)


def entropy(dist):
    # Events with probability 0 do not contribute to the entropy
    dist = dist[dist > 0]
    return -sum([p * math.log(p) for p in dist])


def apply_rna_model(s_dist, context, model, cache, r_threshold, s_threshold):
    if model is None:
        return s_dist

    r_dist = get_next_base_prob(context, model, cache)

    # combine the probability distributions from the RNA and sig2seq models
    r_entropy = entropy(r_dist) # TODO use cache for improved efficiency (key: context, value: entropy of pred)
    s_entropy = entropy(normalise(s_dist[:-1]))
    if r_entropy < r_threshold and s_entropy > s_threshold:
        return combine_dists(r_dist, s_dist)
    else:
        return s_dist


# TODO: Define class for decoding params
def beam_search(
    mat: np.ndarray,
    bases: str,
    beam_width: int,
    lm: tf.keras.Model,
    s_threshold: int,
    r_threshold: int,
    len_context: int,
    cache: dict,
) -> str:
    """Beam search decoder.

    See the paper of Hwang et al. and the paper of Graves et al.

    Args:
        mat: Output of neural network of shape TxC.
        bases: The set of bases the neural network can recognize, excluding the CTC-blank.
        beam_width: Number of beams kept per iteration.
        lm: Character level language model if specified.

    Returns:
        The decoded text.
    """

    blank_idx = len(bases)
    max_T, max_C = mat.shape

    # initialise beam state
    last = BeamList()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].pr_blank = log(1)
    last.entries[labeling].pr_total = log(1)

    # go over all time-steps
    for t in range(max_T):
        # TODO: Remove (test to skip confusion at start of matrix)
        if t < 30:
            continue
        curr = BeamList()

        # get beam-labelings of best beams
        best_labelings = last.sort_labelings()[0][:beam_width]

        # go over best beams
        for labeling in best_labelings:

            # COPY BEAM

            # probability of paths ending with a non-blank
            pr_non_blank = log(0)
            # in case of non-empty beam
            if labeling:
                # apply RNA model to the posteriors
                if len(labeling) >= len_context + 1:
                    context = get_context(labeling, len_context, exclude_last=True)
                    pr_dist = apply_rna_model(mat[t], context, lm, cache, r_threshold, s_threshold)
                else:
                    pr_dist = mat[t]

                pr_non_blank = last.entries[labeling].pr_non_blank + log(pr_dist[labeling[-1]])

            # probability of paths ending with a blank
            pr_blank = last.entries[labeling].pr_total + log(mat[t, blank_idx])

            # fill in data for current beam
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].pr_non_blank = np.logaddexp(curr.entries[labeling].pr_non_blank, pr_non_blank)
            curr.entries[labeling].pr_blank = np.logaddexp(curr.entries[labeling].pr_blank, pr_blank)
            curr.entries[labeling].pr_total = np.logaddexp(curr.entries[labeling].pr_total,
                                                           np.logaddexp(pr_blank, pr_non_blank))
            curr.entries[labeling].indices = last.entries[labeling].indices

            # EXTEND BEAM

            # apply RNA model to the posteriors
            if len(labeling) >= len_context:
                context = get_context(labeling, len_context, exclude_last=False)
                pr_dist = apply_rna_model(mat[t], context, lm, cache, r_threshold, s_threshold)
            else:
                pr_dist = mat[t]

            # extend current beam-labeling
            for c in range(max_C - 1):
                # add new char to current beam-labeling
                new_labeling = labeling + (c,)

                # keep track of matrix index that new char is located at
                new_indices = curr.entries[labeling].indices + (t,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    pr_non_blank = last.entries[labeling].pr_blank + log(pr_dist[c])
                else:
                    pr_non_blank = last.entries[labeling].pr_total + log(pr_dist[c])

                # fill in data TODO: Refactor
                curr.entries[new_labeling].labeling = new_labeling
                curr.entries[new_labeling].pr_non_blank = np.logaddexp(curr.entries[new_labeling].pr_non_blank,
                                                                       pr_non_blank)
                curr.entries[new_labeling].pr_total = np.logaddexp(curr.entries[new_labeling].pr_total, pr_non_blank)
                curr.entries[new_labeling].indices = new_indices

        # set new beam state
        last = curr

    # sort by probability
    sorted_labels = last.sort_labelings()
    best_labeling = sorted_labels[0][0]  # get most probable labeling
    indices = sorted_labels[1][0]

    # map label string to sequence of bases
    best_seq = ''.join([bases[label] for label in best_labeling])

    return best_seq, indices