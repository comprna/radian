import glob

import pandas as pd

def main():
    tsv_dir = "/mnt/sda/rna-basecaller/experiments/with-rna-model/global/all_val/investigate_softmax/reprocessed"

    for file in glob.glob(f"{tsv_dir}/*.tsv"):

        read = file.split(f"{tsv_dir}/")[1].split("_")[0]
        print(f"Read: {read}")

        # Load the beams

        df = pd.read_csv(file, sep="\t", names=['read', 'k', 'gt', 'pred', 'ed'])

        # Which beam has the lowest edit distance and what is it?

        min_rows = df[df.ed == df.ed.min()]
        min_beams = []
        min_ed = None
        for i, row in min_rows.iterrows():
            min_beams.append(row['k'])
            min_ed = row['ed']
        print(f"Beams {min_beams} have lowest ED of {min_ed:.3f}")


        # Which beam has the highest edit distance and what is it?

        max_rows = df[df.ed == df.ed.max()]
        max_beams = []
        max_ed = None
        for i, row in max_rows.iterrows():
            max_beams.append(row['k'])
            max_ed = row['ed']
        print(f"Beams {max_beams} have highest ED of {max_ed:.3f}")


        # What edit distance ranking does the first beam have?

        ed = df.iloc[0]['ed']
        sorted_df = df.sort_values(by=['ed'], ignore_index=True)
        rank = sorted_df.index[sorted_df.k == 1].tolist()[0]
        print(f"The top beam has ED {ed:.3f} which is {rank}th best\n\n")




if __name__ == "__main__":
    main()