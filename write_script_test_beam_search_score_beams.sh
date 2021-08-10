#!/bin/bash

i=7

for normalise_after in 0 1
do
    for lm_factor in 0.2 0.6
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

module load cuda/10.1
module load intel-mkl/2019.3.199
module load tensorflow/2.3.0
module load python3/3.7.4

source /home/150/as2781/rnabasecaller/.venv/bin/activate

SCRIPT='/home/150/as2781/rnabasecaller/test_beam_search_score_beams_gadi.py'
MODEL_DIR='/g/data/xc17/Eyras/alex/working/with-rna-model/global/all_val/copied_files/models'
R_CONFIG='\$MODEL_DIR/r-config-37.yaml'
R_MODEL='\$MODEL_DIR/r-train-37-model-03.h5'
CONTEXT_LEN = 8

python3 \$SCRIPT \$R_CONFIG \$R_MODEL \$CONTEXT_LEN $beam_width $lm_factor $normalise_after

            " > decode-$i.sh

            let "i += 1"
        done
    done
done
