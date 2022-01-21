#!/bin/bash


source activate snn_gpu
module load cuda
echo `which python`
cd $PRODUCTS/classifiers/supernnova

echo "#################TIMING  Starting here:   `date`"

# database
python run.py --data --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --redshift_label REDSHIFT_FINAL --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES --raw_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/1_SIM/26XBOOSTEDDES/PIP_AM_DES5YR_SIMS_26XBOOSTEDDES

#seeds as in paper
for seed in 0 100 1000 55 30469 30496 49 7510 1444 9 1 2 303 4444 537
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES --model vanilla  --redshift zspe --norm cosmo  --train_rnn --seed $seed --num_layers 4 --hidden_dim 64
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES --model vanilla  --redshift zspe --norm cosmo_quantile  --train_rnn --seed $seed --num_layers 4 --hidden_dim 64
done
