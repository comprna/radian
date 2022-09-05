import os
from pathlib import Path

import pandas as pd

def main():
    # Results directory

    bw            = 30
    approach      = "dynamic"
    results_dir   = f"./decode_out/{approach}/{bw}"

    # Load baseline

    baseline_dir  = "./decode_out/no_rna_model"
    baseline_file = "decode-7-out.txt" if bw == 30 else "decode-8-out.txt"
    baseline      = pd.read_csv(f"{baseline_dir}/{baseline_file}",
                                sep='\t',
                                names=['i','read_id','gt','pred','ed'])
 
    # Summarise results
    
    summary = []
    for file in sorted(Path(results_dir).iterdir(), key=os.path.getctime):
        
        # Get run number

        run = file.name.split('.')[0].split('-out')[0]

        # Load results

        df = pd.read_csv(file, sep='\t', names=['i','read_id','gt','pred','ed'], comment='#')

        # Merge with baseline for comparison

        df = pd.merge(df, baseline, on="read_id")

        # Subset columns with model and baseline results

        df = df[['i_x', 'read_id', 'pred_x', 'ed_x', 'pred_y', 'ed_y']]
        df = df.rename(columns={"i_x": "i_m", "pred_x": "pred_m",
                                "ed_x": "ed_m", "pred_y": "pred_b",
                                "ed_y": "ed_b"})

        # Add better / worse / same results

        def get_change(row):
            ed_m = row['ed_m']
            ed_b = row['ed_b']
            if row['ed_m'] < row['ed_b']:
                return "better"
            elif row['ed_m'] > row['ed_b']:
                return "worse"
            else:
                return "same"
        df['change'] = df.apply(lambda row: get_change(row), axis=1)

        # Summarise results

        b_ed      = df['ed_b'].mean()  
        m_ed      = df['ed_m'].mean()
        change_ed = (m_ed - b_ed) / b_ed * 100

        n_reads   = len(df)
        n_better  = len(df[df['change'] == 'better'])
        n_worse   = len(df[df['change'] == 'worse'])
        n_same    = len(df[df['change'] == 'same'])

        p_better  = n_better / n_reads * 100
        p_worse   = n_worse / n_reads * 100
        p_same    = n_same / n_reads * 100

        # Add to summary table

        summary.append([run, n_reads, b_ed, m_ed, change_ed, n_better, 
                        p_better, n_same, p_same, n_worse, p_worse])

    df = pd.DataFrame.from_records(summary, columns=[
        'run', 'n_reads', 'b_ed', 'm_ed', 'change_ed', 'n_better', 'p_better',
        'n_same', 'p_same', 'n_worse', 'p_worse'])

    df.to_csv(f"{results_dir}/summary.tsv", sep="\t")

if __name__ == "__main__":
    main()