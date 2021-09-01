import json
import random
from statistics import mean

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import numpy as np
import pandas as pd
from textdistance import levenshtein


def homopolymer_at(start, end, gt, alignment, candidate):
    # We consider the deletion to be in a homopolymer if the deleted
    # position in the ground truth is at the start, middle or end of a
    # homopolymer of length >= 3.  There can be no insertions or
    # substitutions within the homopolymer, otherwise the prediction is
    # not a shortened version of the homopolymer.

    # First check that start and end are valid positions.
    if start < 0:
        return False
    
    if end > len(alignment) - 1:
        return False

    # Note that the first condition will be false if there is an insertion, 
    # so we only need to check for substitutions in the second condition.
    return gt[start:end+1] == candidate and '.' not in alignment[start:end+1]


def is_homopolymer_deletion(pos, alignment, gt):
    b = gt[pos]
    candidate = b + b + b

    if homopolymer_at(pos-2, pos, gt, alignment, candidate) or \
        homopolymer_at(pos-1, pos+1, gt, alignment, candidate) or \
            homopolymer_at(pos, pos+2, gt, alignment, candidate):
        return True

    return False


def is_sub(a, b, gt, pred):
    if gt == a and pred == b or gt == b and pred == a:
        return True
    return False


def analyse_alignment(formatted_alignment):

    # Metrics of interest

    n_sub   = 0  # Total number of substitutions
    n_ins   = 0  # Total number of insertions
    n_del   = 0  # Total number of deletions
    n_hdel  = 0  # Deletions in homopolymers
    n_ctsub = 0  # Substitutions of C into T or T into C
    n_cgsub = 0  # Substitutions of C into G or G into C
    n_casub = 0  # Substitutions of C into A or A into C
    n_gasub = 0  # Substitutions of G into A or A into G
    n_gtsub = 0  # Substitutions of G into T or T into T
    n_atsub = 0  # Substitutions of A into T or T into A

    # Break down formatted alignment

    bases     = {'A', 'C', 'G', 'T'}
    lines     = formatted_alignment.split('\n')
    gt        = lines[0]
    alignment = lines[1]
    pred      = lines[2]

    assert(len(gt)   == len(alignment))
    assert(len(pred) == len(alignment))

    # Parse alignment string to determine error types

    for i in range(len(alignment)):

        # Correct alignment

        if alignment[i] == "|":
            continue
        
        # Substitution

        elif alignment[i] == '.':
            n_sub += 1

            if is_sub('C', 'T', pred[i], gt[i]):
                n_ctsub += 1
            elif is_sub('C', 'G', pred[i], gt[i]):
                n_cgsub += 1
            elif is_sub('C', 'A', pred[i], gt[i]):
                n_casub += 1
            elif is_sub('G', 'A', pred[i], gt[i]):
                n_gasub += 1
            elif is_sub('G', 'T', pred[i], gt[i]):
                n_gtsub += 1
            elif is_sub('A', 'T', pred[i], gt[i]):
                n_atsub += 1

        elif alignment[i] == ' ':

            # Deletion

            if gt[i] in bases:
                n_del += 1
                
                # Check if in homopolymer

                if is_homopolymer_deletion(i, alignment, gt):
                    n_hdel += 1
           
            # Insertion

            elif pred[i] in bases:
                n_ins += 1
    
    return { 'n_sub':   n_sub,
             'n_ins':   n_ins,
             'n_del':   n_del,
             'n_hdel':  n_hdel,
             'n_ctsub': n_ctsub,
             'n_cgsub': n_cgsub,
             'n_casub': n_casub, 
             'n_gasub': n_gasub,
             'n_gtsub': n_gtsub,
             'n_atsub': n_atsub }


def main():

    # Write read errors to tsv file

    print("read\tgt_length\tn_alignments\tn_sub\tn_ins\tn_del\tn_hdel\tn_ctsub\tn_cgsub\tn_casub\tn_gasub\tn_gtsub\tn_atsub")
    
    results_file = './decode_out/dynamic/30/decode-51-out.txt'
    with open(results_file, "r") as f:

        for line in f:
            results = line.split('\t')
            read    = results[1]
            gt      = results[2]
            pred    = results[3]
    
            # Pick a random alignment to analyse

            alignments = pairwise2.align.globalms(gt, pred, 2, -4, -4, -2)
            alignment = random.choice(alignments)

            # Analyse errors in alignment

            errors = analyse_alignment(format_alignment(*alignment))
            print(f"{read}\t{len(gt)}\t{len(alignments)}\t{errors['n_sub']}\t{errors['n_ins']}\t{errors['n_del']}\t{errors['n_hdel']}\t{errors['n_ctsub']}\t{errors['n_cgsub']}\t{errors['n_casub']}\t{errors['n_gasub']}\t{errors['n_gtsub']}\t{errors['n_atsub']}")


if __name__ == "__main__":
    main()