import random
import sys

from Bio import pairwise2
from Bio import SeqIO
from Bio.pairwise2 import format_alignment
import numpy as np

def analyse_alignment(formatted_alignment):
    # Metrics of interest
    n_mat   = 0  # Number of matches
    n_sub   = 0  # Total number of substitutions
    n_ins   = 0  # Total number of insertions
    n_del   = 0  # Total number of deletions

    # Break down formatted alignment
    bases     = {'A', 'C', 'G', 'T'}
    lines     = formatted_alignment.split('\n')
    gt        = lines[0]
    alignment = lines[1]
    pred      = lines[2]
    assert(len(gt)   == len(alignment))
    assert(len(pred) == len(alignment))

    # Simple implementation of soft clipping to remove remove garbage from 
    # start and end of query sequence.  Considers real sequence to start once
    # there are > 2 non-insertions in a row.
    # Determine clipping at start of query sequence
    clip_start = 0
    for i in range(len(gt)):
        clip_start = i
        if gt[i] != '-' and gt[i+1] != '-' and gt[i+2] != '-': # Index error will alert us to erroneous alignment
            break
    # Determine clipping at end of query sequence
    clip_end = len(alignment) - 1
    for i in range(len(gt)-1, -1, -1):
        clip_end = i
        if gt[i] != '-' and gt[i-1] != '-' and gt[i-2] != '-': # Index error will alert us to erroneous alignment
            break

    alignment = alignment[clip_start:clip_end+1]
    gt = gt[clip_start:clip_end+1]
    pred = pred[clip_start:clip_end+1]

    # Parse alignment string to determine error types
    for i in range(len(alignment)):
        if alignment[i] == "|":       # Match
            n_mat += 1
        elif alignment[i] == '.':     # Substitution
            n_sub += 1
        elif alignment[i] == ' ':
            if gt[i] in bases:        # Deletion
                n_del += 1
            elif pred[i] in bases:    # Insertion
                n_ins += 1

    return n_mat, n_sub, n_ins, n_del

def main():
    # fasta = "/mnt/sda/rna-basecaller/experiments/decode/global-n-gram/5_IndepTest/fasta/guppy-gm12878.fastq"
    fasta = sys.argv[1]
    # ref = "/mnt/sda/rna-basecaller/experiments/decode/global-n-gram/4_Align/read_ref_seq_gm12878.tsv"
    ref = sys.argv[2]
    out_file = fasta.replace(".fasta", ".tsv")

    # Load reads + ref seqs
    read_ref = {}
    with open(ref, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue # Skip header
            read, txt, seq = line.strip('\n').split('\t')
            read_ref[read] = seq

    # Parse fasta file and align
    stats = []
    with open(out_file, "w") as out:
        out.write("read_id\tn_match\tn_ins\tn_del\tn_sub\n")
        for seq_record in SeqIO.parse(fasta,  "fasta"):
            read = seq_record.id
            seq = str(seq_record.seq)
            ref = read_ref[read]

            # Since ref contains Ts, make sure basecaller calls T instead of U
            seq = seq.replace("U", "T")

            # Align using same parameters as minimap2
            alignments = pairwise2.align.globalms(ref, seq, 2, -4, -4, -2)
            alignment = random.choice(alignments)

            # Compute accuracy, errors
            n_match, n_sub, n_ins, n_del = analyse_alignment(format_alignment(*alignment))
            acc = n_match / (n_match + n_sub + n_ins + n_del) * 100
            p_ins = n_ins / (n_match + n_sub + n_ins + n_del) * 100
            p_del = n_del / (n_match + n_sub + n_ins + n_del) * 100
            p_sub = n_sub / (n_match + n_sub + n_ins + n_del) * 100
            p_err = (n_ins + n_del + n_sub) / (n_match + n_sub + n_ins + n_del) * 100 
            stats.append([acc, p_ins, p_del, p_sub, p_err])

            # Write raw stats for downstream analysis
            out.write(f"{read}\t{n_match}\t{n_ins}\t{n_del}\t{n_sub}\n")

    # Print metrics
    stats = np.asarray(stats)
    print(f"Accuracy\tMEDIAN: {np.median(stats[:,0]):.2f}\tMEAN: {np.mean(stats[:,0]):.2f}")
    print(f"Insertions\tMEDIAN: {np.median(stats[:,1]):.2f}\tMEAN: {np.mean(stats[:,1]):.2f}")
    print(f"Deletions\tMEDIAN: {np.median(stats[:,2]):.2f}\tMEAN: {np.mean(stats[:,2]):.2f}")
    print(f"Substitutions\tMEDIAN: {np.median(stats[:,3]):.2f}\tMEAN: {np.mean(stats[:,3]):.2f}\n")
    print(f"Total error\tMEDIAN: {np.median(stats[:,4]):.2f}\tMEAN: {np.mean(stats[:,4]):.2f}\n")


if __name__ == "__main__":
    main()