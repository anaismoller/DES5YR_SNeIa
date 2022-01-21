#!/bin/bash

source activate snn_gpu
module load cuda
echo `which python`
# Beware, batch size of database may need reduction
cd $PRODUCTS/classifiers/supernnova

echo "#################TIMING  Starting here:   `date`"


for seed in 0 100 1000 55 30469 30496 49 7510 1444 9
do
  echo $seed
  # J17
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir /$PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_J17 --model vanilla  --redshift zspe --norm cosmo  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files /$PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean.pt
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir /$PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_J17 --model vanilla  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files /$PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean.pt

  # PSNID
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir /$PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_PSNID --model vanilla  --redshift zspe --norm cosmo  --validate_rnn--seed $seed --num_layers 4 --hidden_dim 64 --model_files /$PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean.pt
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir /$PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_PSNID --model vanilla  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files /$PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean.pt

done
