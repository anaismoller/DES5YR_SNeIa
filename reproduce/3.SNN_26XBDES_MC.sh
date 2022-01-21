#!/bin/bash


source activate snn_gpu
module load cuda
echo `which python`
cd $PRODUCTS/classifiers/supernnova

echo "#################TIMING  Starting here:   `date`"


for seed in 0 100 1000 55 30469 30496 49 7510 1444 99 1 2 303 4444 537
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES --batch_size 1024  --redshift zspe --norm cosmo_quantile  --train_rnn --num_layers 4 --hidden_dim 64 --model variational --seed $seed 
done

