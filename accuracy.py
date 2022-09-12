import re
import sys

import numpy as np
import pysam


def main():
    # SAM file to parse
    sam_file = sys.argv[1]
    # sam_file = "/home/alex/Documents/tmp/ngram-1-aln.sam"
    out_file = sam_file.replace(".sam", ".tsv")

    # Store stats for all reads
    stats = []

    # Parse SAM file
    sam_file = pysam.AlignmentFile(sam_file, "r")
    with open(out_file, "w") as out:
        out.write("read_id\tref_name\tn_match\tn_ins\tn_del\tn_sub\n")
        
        n_unmapped = 0
        for read in sam_file:
            if read.is_unmapped:
                n_unmapped += 1
                continue

            if not read.seq:
                print("ERROR: NO QUERY SEQ")
                continue

            # Extract relevant info from alignment record
            read_id = read.qname
            ref_name = read.reference_name.split("|")
            transcript = ref_name[0]

            # Calculate metrics
            n_match = 0
            n_ins = 0
            n_del = 0
            n_sub = 0
            for char in read.cigar:
                op = char[0]
                count = char[1]
                if op == 0:            # M (0): Matches + substitutions
                    n_match += count
                elif op == 1:          # I (1): Insertions
                    n_ins += count
                elif op == 2:          # D (2): Deletions
                    n_del += count
            
            nm = read.get_tag("NM")    # NM: Ins + del + sub
            n_sub = nm - n_ins - n_del
            n_match -= n_sub

            # Write raw stats for downstream analysis
            out.write(f"{read_id}\t{transcript}\t{n_match}\t{n_ins}\t{n_del}\t{n_sub}")

            # Compute accuracy and error percentages
            acc = n_match / (n_match + nm) * 100
            p_ins = n_ins / (n_match + nm) * 100
            p_del = n_del / (n_match + nm) * 100
            p_sub = n_sub / (n_match + nm) * 100
            p_err = (n_ins + n_del + n_sub) / (n_match + nm) * 100
            stats.append([acc, p_ins, p_del, p_sub, p_err])

    # Print metrics
    stats = np.asarray(stats)
    print(f"N unmapped reads: {n_unmapped}")
    print(f"N mapped reads: {len(stats)}")
    print(f"Accuracy\tMEDIAN: {np.median(stats[:,0]):.2f}\tMEAN: {np.mean(stats[:,0]):.2f}")
    print(f"Insertions\tMEDIAN: {np.median(stats[:,1]):.2f}\tMEAN: {np.mean(stats[:,1]):.2f}")
    print(f"Deletions\tMEDIAN: {np.median(stats[:,2]):.2f}\tMEAN: {np.mean(stats[:,2]):.2f}")
    print(f"Substitutions\tMEDIAN: {np.median(stats[:,3]):.2f}\tMEAN: {np.mean(stats[:,3]):.2f}\n")
    print(f"Total error\tMEDIAN: {np.median(stats[:,4]):.2f}\tMEAN: {np.mean(stats[:,4]):.2f}\n")


if __name__ == "__main__":
    main()
