#!/bin/bash

i=21

for r_threshold in 1000 0.9 0.4
do
    for s_threshold in 0 0.4 0.9
    do
        for beam_width in 30 100
        do

            echo "
#!/bin/bash
#PBS -P xc17
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=10GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -M Alexandra.Sneddon@anu.edu.au
#PBS -m abe
#PBS -j oe
#PBS -l storage=scratch/xc17+gdata/xc17

module load tensorflow/2.3.0
module load python3/3.8.5

source /home/150/as2781/rnabasecaller/.venv/bin/activate

SCRIPT='/home/150/as2781/rnabasecaller/test_beam_search_dynamic_gadi.py'
MODEL_DIR='/g/data/xc17/Eyras/alex/working/with-rna-model/global/all_val/copied_files/models'
R_CONFIG="\$MODEL_DIR/r-config-37.yaml"
R_MODEL="\$MODEL_DIR/r-train-37-model-03.h5"
CONTEXT_LEN=8

python3 \$SCRIPT \$R_CONFIG \$R_MODEL \$CONTEXT_LEN $beam_width $r_threshold $s_threshold

            " > decode-$i.sh

            let "i += 1"
        done
    done
done
