from os import confstr
import pandas as pd

def main():

    # Load TCN EDs

    tcn_file = '/mnt/sda/rna-basecaller/experiments/decode/without-rna-model/benchmark/3_OurTcnBasecall/decode-7-all-hek293-out.tsv'
    tcn_df = pd.read_csv(tcn_file, sep='\t', names=['i', 'read_id', 'gt_tcn', 'pred_tcn', 'ed_tcn'])
    tcn_df = tcn_df[['read_id', 'gt_tcn', 'pred_tcn', 'ed_tcn']]


    # Load Guppy 5 EDs

    guppy5_file = '/mnt/sda/rna-basecaller/experiments/decode/without-rna-model/benchmark/7_CompareEd/write_ed_guppy5_hek293_out.tsv'
    guppy5_df = pd.read_csv(guppy5_file, sep='\t')
    guppy5_df.rename(columns={'read': 'read_id', 'gt':'gt_guppy', 'pred':'pred_guppy', 'ed':'ed_guppy'}, inplace=True)


    # Merge results for comparison

    merged_df = pd.merge(tcn_df, guppy5_df, on="read_id")


    # Check that the GT values are the same (so ED has been calculated correctly)

    conflicts = merged_df[merged_df['gt_tcn'] != merged_df['gt_guppy']]
    assert len(conflicts.index) == 0


    # Calculate average ED with TCN and guppy

    tcn_mean = merged_df['ed_tcn'].mean()
    guppy_mean = merged_df['ed_guppy'].mean()
    print(f"TCN ED mean: {tcn_mean}")
    print(f"Gup ED mean: {guppy_mean}")


    # Calculate number of predictions where TCN is better than guppy

    better = merged_df[merged_df['ed_tcn'] < merged_df['ed_guppy']]
    n_better = len(better.index)
    print(f"N better: {n_better}")
    print(f"% better: {n_better/len(merged_df.index)*100}")


    # Calculate number of predictions where TCN is worse than guppy

    worse = merged_df[merged_df['ed_tcn'] > merged_df['ed_guppy']]
    n_worse = len(worse.index)
    print(f"N worse: {n_worse}")
    print(f"% worse: {n_worse/len(merged_df.index)*100}")


    # Calculate number of predictions where TCN is same as guppy

    same = merged_df[merged_df['ed_tcn'] == merged_df['ed_guppy']]
    n_same = len(same.index)
    print(f"N same: {n_same}")
    print(f"% same: {n_same/len(merged_df.index)*100}")



if __name__ == "__main__":
    main()