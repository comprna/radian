import sys

def main():

    fastq_file = sys.argv[1]

    with open(fastq_file, "r") as f:
        lines = [ line.rstrip('\n') for line in f ]
        
    # Reads are in groups of 4 lines
    
    read_info = lines[0::4]  # read id and run id
    pred      = lines[1::4]  # predicted sequence
    strand    = lines[2::4]  # strand
    phred     = lines[3::4]  # phred score

    # Extract read IDs

    reads = [ info.split(' ')[0][1:] for info in read_info ]

    # Write read IDs and preds to file
    print(f"read\tpred")
    for i, read in enumerate(reads):
        print(f"{read}\t{pred[i]}")


if __name__ == "__main__":
    main()