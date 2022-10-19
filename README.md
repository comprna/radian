# RADIAN

**R**NA l**A**nguage informe**D** decod**I**ng of n**A**nopore sig**N**als

# Overview

Nanopore direct RNA basecaller that utilises a model of mRNA language.

Since RNA is always sequenced from the 3' to 5' direction, nanopore signals implicitly encode the nucleotide biases in mRNA.  This basecaller uses a probabilistic model of mRNA language to guide basecalling when the signal prediction is ambiguous.  The mRNA model is incorporated in a modified CTC beam search decoding algorithm.

![RADIAN architecture](arch.png?raw=true)


# Installation

```
cd <path/to/radian>
pip install --upgrade pip
pip install -r requirements.txt
```

# Command structure

```
usage: basecall.py [-h] fast5_dir fasta_dir [--local] [--chunk-len] [--step-size]
                   [--batch-size] [--outlier-clip] [--rna-model]
                   [--sig-model] [--sig-config] [--beam-width]
                   [--decode-type] [--sig-threshold]
                   [--rna-threshold] [--context-len]

positional arguments:
  fast5_dir             Directory of single/multi fast5 files.
  fasta_dir             Directory to output fasta files.

optional arguments:
  -h, --help
  --local
  --chunk-len
  --step-size
  --batch-size
  --outlier-clip
  --rna-model
  --sig-model
  --sig-config
  --beam-width
  --decode-type {global,chunk}
  --sig-threshold
  --rna-threshold
  --context-len
```

# Example usage

We provide a fast5 file containing 5 reads for testing in data/reads.fast5.

To basecall the single or multi-fast5 file(s) in <data> and output fasta to <out_dir>:
```
cd radian
mkdir out_dir
python3 basecall.py data out_dir
```
