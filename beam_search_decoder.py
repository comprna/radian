"""
This has been adapted from https://github.com/githubharald/CTCDecoder/blob/master/ctc_decoder/beam_search.py
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import normalize
import tensorflow as tf


N_BASES = 4

def log(x: float) -> float:
    with np.errstate(divide='ignore'):
        return np.log(x)


@dataclass
class BeamEntry:
    """Information about one single beam at specific time-step."""
    pr_total: float = log(0)  # blank and non-blank
    pr_non_blank: float = log(0)  # non-blank
    pr_blank: float = log(0)  # blank
    pr_text: float = log(1)  # LM score
    lm_applied: bool = False  # flag if LM was already applied to this beam
    labeling: tuple = ()  # beam-labeling
    indices: tuple = ()  # indices of each character in beam


class BeamList:
    """Information about all beams at specific time-step."""

    def __init__(self) -> None:
        self.entries = defaultdict(BeamEntry)

    def normalize(self) -> None:
        """Length-normalise LM score."""
        for k in self.entries.keys():
            labeling_len = len(self.entries[k].labeling)
            self.entries[k].pr_text = (1.0 / (labeling_len if labeling_len else 1.0)) * self.entries[k].pr_text

    def sort_labelings(self) -> List[Tuple[int]]:
        """Return beam-labelings, sorted by probability."""
        beams = self.entries.values()
        sorted_beams = sorted(beams, reverse=True, key=lambda x: x.pr_total + x.pr_text)
        return [x.labeling for x in sorted_beams], [x.indices for x in sorted_beams]

def apply_rna_model_to_seq(
    parent_beam: BeamEntry,
    child_beam: BeamEntry,
    lm: tf.keras.Model,
    cache: dict,
    factor: int,
    len_context: int
) -> None:
    """Calculate LM score of child beam by taking score from parent beam and RNA model probability."""
    if not lm or child_beam.lm_applied:
        return
    
    if len(parent_beam.labeling) < len_context:
        return

    context = parent_beam.labeling[-len_context:]
    if context not in cache:
        context = tf.one_hot(list(parent_beam.labeling[-len_context:]), N_BASES).numpy()
        context = context.reshape(1, len_context, N_BASES)
        probs = lm.predict(context)
        cache[context] = probs
    else:
        probs = cache[context]

    new_char = child_beam.labeling[-1]
    child_beam.pr_text = parent_beam.pr_text + factor * log(probs[0][new_char])  # probability of sequence
    child_beam.lm_applied = True  # only apply LM once per beam entry

def get_context(labeling, len_context, exclude_last=False):
    # the context is the last portion of the beam
    if exclude_last == True:
        context = labeling[-(len_context+1):-1]
    else:
        context = labeling[-len_context:]

    return context

def apply_rna_model(s_dist, context, model, cache, threshold):
    if model is None:
        return s_dist

    # update the cache if needed
    if context not in cache:
        # convert context into RNA model input format
        context_arr = tf.one_hot(list(context), N_BASES).numpy()
        context_arr = context_arr.reshape(1, len(context), N_BASES)

        # predict the distribution of the next base given the context
        r_dist = model.predict(context_arr)[0]
        cache[context] = r_dist
    else:
        r_dist = cache[context]

    r_entropy = entropy(r_dist)
    s_entropy = entropy(s_dist[:-1]) # exclude the blank from the entropy calc

    # combine the probability distributions from the RNA and sig2seq models
    if r_entropy < threshold:
        
        s_base_prob = np.sum(s_dist[:-1])
        s_base_dist = s_dist[:-1] / s_base_prob
        # alter the signal probs according to the rna model probs 

        s_A = (s_base_dist[0] + r_dist[0]) / 2

        s_dist_mod = np.add(r_dist, s_base_dist)
        s_dist_mod = normalize([s_dist_mod], norm="l1")[0]
        s_dist_mod = s_dist_mod * s_base_prob
        s_dist_mod = np.append(s_dist_mod, s_dist[-1])

        return s_dist_mod
    else:
        return s_dist

# TODO: Define class for decoding params
def beam_search(
    mat: np.ndarray,
    bases: str,
    beam_width: int,
    lm: tf.keras.Model,
    factor: int,
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

        # get the entropy of the current timestep
        dist = mat[t]
        t_entropy = entropy(mat[t])

        # go over best beams
        for labeling in best_labelings:

            # apply RNA model to the posteriors
            # TODO: test t_entropy condition inside apply_rna_model
            if len(labeling) >= len_context and t_entropy > s_threshold:
                context = get_context(labeling, len_context, exclude_last=False)
                pr_dist = apply_rna_model(mat[t], context, lm, cache, r_threshold)
            else:
                pr_dist = mat[t]

            # COPY BEAM: https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7

            # probability of paths ending with a non-blank
            pr_non_blank = log(0)
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                c = labeling[-1]
                # TODO: Update (note that RNA model prob above was calculated using incorrect beam)
                pr_char = log(mat[t, c])

                # apply RNA model
                # if len(labeling) >= len_context+1 and t_entropy > s_threshold:
                #     context = get_context(labeling, len_context, exclude_last=True)
                #     pr_char = apply_rna_model(pr_char, c, context, lm, cache, r_threshold)

                pr_non_blank = last.entries[labeling].pr_non_blank + pr_char

            # probability of paths ending with a blank
            pr_blank = last.entries[labeling].pr_total + log(mat[t, blank_idx])

            # fill in data for current beam
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].pr_non_blank = np.logaddexp(curr.entries[labeling].pr_non_blank, pr_non_blank)
            curr.entries[labeling].pr_blank = np.logaddexp(curr.entries[labeling].pr_blank, pr_blank)
            curr.entries[labeling].pr_total = np.logaddexp(curr.entries[labeling].pr_total,
                                                           np.logaddexp(pr_blank, pr_non_blank))
            curr.entries[labeling].pr_text = last.entries[labeling].pr_text
            curr.entries[labeling].lm_applied = True  # LM already applied at previous time-step for this beam-labeling
            curr.entries[labeling].indices = last.entries[labeling].indices

            # EXTEND BEAM

            # extend current beam-labeling
            for c in range(max_C - 1):
                # add new char to current beam-labeling
                new_labeling = labeling + (c,)

                # keep track of matrix index that new char is located at
                new_indices = curr.entries[labeling].indices + (t,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    pr_non_blank = last.entries[labeling].pr_blank + pr_dist[c]
                else:
                    pr_non_blank = last.entries[labeling].pr_total + pr_dist[c]

                # fill in data TODO: Refactor
                curr.entries[new_labeling].labeling = new_labeling
                curr.entries[new_labeling].pr_non_blank = np.logaddexp(curr.entries[new_labeling].pr_non_blank,
                                                                       pr_non_blank)
                curr.entries[new_labeling].pr_total = np.logaddexp(curr.entries[new_labeling].pr_total, pr_non_blank)
                curr.entries[new_labeling].indices = new_indices

        # set new beam state
        last = curr

    # TODO: Remove, not needed with new approach
    # normalise LM scores according to beam-labeling-length
    last.normalize()

    # sort by probability
    sorted_labels = last.sort_labelings()
    best_labeling = sorted_labels[0][0]  # get most probable labeling
    indices = sorted_labels[1][0]

    # map label string to sequence of bases
    best_seq = ''.join([bases[label] for label in best_labeling])

    return best_seq, indices