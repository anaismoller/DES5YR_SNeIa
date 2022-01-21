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
  # bayesian
  # J17
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1050 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_14X_J17/ --model bayesian  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_J17/models/bayesian_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1050_True_mean_Bayes_0.75_-1.0_-7.0_4.0_3.0_-1.0_-7.0_4.0_3.0/bayesian_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1050_True_mean_Bayes_0.75_-1.0_-7.0_4.0_3.0_-1.0_-7.0_4.0_3.0.pt --num_inference_samples 10
  # PSNID
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1050 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_14X_PSNID/ --model bayesian  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_PSNID/models/bayesian_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1050_True_mean_Bayes_0.75_-1.0_-7.0_4.0_3.0_-1.0_-7.0_4.0_3.0/bayesian_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1050_True_mean_Bayes_0.75_-1.0_-7.0_4.0_3.0_-1.0_-7.0_4.0_3.0.pt --num_inference_samples 10 

  # variational
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1050 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_14X_J17/ --model bayesian  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_J17/models/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1050_True_mean_WD_1e-07/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1050_True_mean_WD_1e-07.pt --num_inference_samples 10
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1050 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_14X_PSNID/ --model bayesian  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_PSNID/models/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1050_True_mean_WD_1e-07/variational_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1050_True_mean_WD_1e-07.pt --num_inference_samples 10
done
