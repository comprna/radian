
def main():

    # Load baseline results

    baseline = {}
    baseline_file = 'decode-7-out.tsv'
    with open(baseline_file, "r") as f:

        for line in f:
            results = line.rstrip('\n').split('\t')
            read    = results[1]
            pred    = results[3]
            ed      = float(results[4])
            baseline[read] = ed

    # Load RNA model results

    rna_model = {}
    results_file = 'decode-15-out.tsv'
    with open(results_file, "r") as f:

        for line in f:
            results = line.rstrip('\n').split('\t')
            read    = results[1]
            pred    = results[3]
            ed      = float(results[4])
            rna_model[read] = ed

    # Combine results

    print("read\tbaseline_ed\tmodel_ed\tchange")
    for read in baseline.keys():
        change = rna_model[read] - baseline[read]
        print(f"{read}\t{baseline[read]}\t{rna_model[read]}\t{change}")


if __name__ == "__main__":
    main()