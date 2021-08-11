from ast import literal_eval as make_tuple

from textdistance import levenshtein

def main():
    tsv_file = "read_0_heart_top_1000_beams.tsv"
    with open(tsv_file, "r") as f:
        lines = [line.rstrip('\n').split("\t") for line in f]
    
    for i, line in enumerate(lines):
        gt = line[2]
        pred_tup = make_tuple(line[3])
        bases = "ACGT"
        pred_seq = ''.join([bases[label] for label in pred_tup])
        ed = levenshtein.normalized_distance(gt, pred_seq)
        print(f"{i}\t{gt}\t{pred_seq}\t{ed}")


if __name__ == "__main__":
    main()