#!/bin/bash

source activate snn_gpu
module load cuda
echo `which python`
# Beware, batch size of database may need reduction
cd $PRODUCTS/classifiers/supernnova

echo "#################TIMING  Starting here:   `date`"

python run.py --data --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --redshift_label REDSHIFT_FINAL --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS_TEST/snndump_5X --raw_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS_TEST/1_SIM/5XDES/PIP_AM_DES5YR_SIMS_TEST_5XDES

for seed in 0 100 1000 55 30469 30496 49 7510 1444 9 1 2 303 4444 537
do
  echo $seed
  # data
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_26XBOOSTEDDES --model variational  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean_WD_1e-07/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean_WD_1e-07.pt
  # DES5X
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS_TEST/snndump_5X --model variational  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean_WD_1e-07/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean_WD_1e-07.pt
  # J17
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_J17/ --model variational  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean_WD_1e-07/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean_WD_1e-07.pt
  # PSNID
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_PSNID/ --model variational  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean_WD_1e-07/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean_WD_1e-07.pt
done
