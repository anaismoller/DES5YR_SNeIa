#!/bin/bash

# Performance of SNN with cosmo and cosmo_quantile norms using Moller et al. 2019 simulations
#
# No GPU?
# comment line "module load cuda"
# eliminate "--use_cuda"

# BEWARE
# need to change ./snn_sims for your directory where downloaded https://zenodo.org/record/3265189#.X85b1C1h3YV

source activate snn_gpu
module load cuda
# $PRODUCTS is where SuperNNova is installed
cd $PRODUCTS/classifiers/supernnova

echo "#################TIMING  Starting here:   `date`"
python run.py --data --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --dump_dir ./dump_SNN_norm --raw_dir ./snn_sims  --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512  --redshift_label REDSHIFT_FINAL
#seeds as in paper
for seed in 0 100 1000 55 30469
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir ./dump_SNN_norm --cyclic --model vanilla   --redshift zspe --norm cosmo  --train_rnn --seed $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir ./dump_SNN_norm --cyclic --model vanilla   --redshift zspe --norm cosmo_quantile  --train_rnn --seed $seed
done

# DF 0.2 for HP comparisson
for seed in 0 100 1000 55 30469
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir ./dump_SNN_norm --cyclic --model vanilla   --redshift zspe --norm cosmo  --train_rnn --seed $seed --data_fraction 0.2
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir ./dump_SNN_norm --cyclic --model vanilla   --redshift zspe --norm cosmo_quantile  --train_rnn --seed $seed --data_fraction 0.2
done