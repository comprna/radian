import glob

def main():
    tsv_dir = "/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/investigate_softmax"

    for file in glob.glob(f"{tsv_dir}/*.tsv"):
        with open(file, "r") as in_f:
            read = file.split(f"{tsv_dir}/")[1].split("_")[0]

            with open(f"{file.split('.tsv')[0]}-mod.tsv", "w") as out_f:

                # Reprocess file to include line breaks
                n_cols = 4
                values = [line.rstrip('\n').split("\t") for line in in_f]
                if len(values) != 1:
                    print(f"Not just 1 line in {in_f}!")
                    break
                for i, val in enumerate(values[0]):
                    if i % n_cols == 0:
                        k = val
                    elif i % n_cols == 1:
                        gt = val
                    elif i % n_cols == 2:
                        pred = val
                    elif i % n_cols == 3:
                        ed = val.split(read)[0]

                        # Now write to file with line breaks
                        out_f.write(f"{read}\t{k}\t{gt}\t{pred}\t{ed}\n")


if __name__ == "__main__":
    main()