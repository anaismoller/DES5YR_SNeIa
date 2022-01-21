#!/bin/bash

# Performance of SNN with cosmo and cosmo_quantile norms using 14XDES simulations
#
# No GPU?
# comment line "module load cuda"
# eliminate "--use_cuda"

# BEWARE
# $PIPPIN_OUTPUT needs to be defined in your configuration

source activate snn_gpu
module load cuda

echo "#################TIMING  Starting here:   `date`"
python run.py --data --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --raw_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS  --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512  --redshift_label REDSHIFT_FINAL
#seeds as in paper
for seed in 0 100 1000 55 30469
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --cyclic --model vanilla  --redshift zspe --norm cosmo  --train_rnn --seed $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --cyclic --model vanilla  --redshift zspe --norm cosmo_quantile  --train_rnn --seed $seed
done
