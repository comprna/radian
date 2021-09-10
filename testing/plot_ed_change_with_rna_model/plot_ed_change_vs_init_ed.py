
import matplotlib.pyplot as plt
import pandas as pd

def main():

    df = pd.read_csv('rna_model_ed_change_decode_7_15.tsv', sep='\t')
    print(df)

    correlation = df['change'].corr(df['baseline_ed'])
    print(correlation)

    df.plot.scatter(x='baseline_ed', y='change')
    plt.title("Plot of ED without model versus ED change with model")
    plt.xlabel("Baseline ED")
    plt.ylabel("ED Change with RNA Model")
    plt.show()


if __name__ == "__main__":
    main()