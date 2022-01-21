#!/bin/bash

source activate snn_gpu
module load cuda
echo `which python`
cd $PRODUCTS/classifiers/supernnova

echo "#################TIMING  Starting here:   `date`"

# database
python run.py --data --data_testing --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --redshift_label REDSHIFT_FINAL --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_26XBOOSTEDDES --raw_dir $DES_ROOT/lcmerge/DESALL_forcePhoto_real_snana_fits --photo_window_files $PIPPIN_OUTPUT/AM_DATA5YR/0_DATAPREP/DATA5YR_DENSE_SNR/DESALL_forcePhoto_real_snana_fits.SNANA.TEXT  --fits_dir $PIPPIN_OUTPUT/AM_DATA5YR/2_LCFIT/D_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz

# with redshift
for seed in 0 100 1000 55 30469 30496 49 7510 1444 9 1 2 303 4444 537
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_26XBOOSTEDDES --model vanilla --norm cosmo  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/vanilla_S_${seed}_CLF_2_R_none_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean/vanilla_S_${seed}_CLF_2_R_none_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean.pt
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_26XBOOSTEDDES --model vanilla --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/vanilla_S_${seed}_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean/vanilla_S_${seed}_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean.pt
done

# without redshift
for seed in 0 100 1000 55 30469 30496 49 7510 1444 9 1 2 303 4444 537
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_26XBOOSTEDDES --model vanilla --norm cosmo  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/vanilla_S_${seed}_CLF_2_R_none_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean/vanilla_S_${seed}_CLF_2_R_none_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean.pt
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DATA5YR/snndump_26XBOOSTEDDES --model vanilla --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/vanilla_S_${seed}_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean/vanilla_S_${seed}_CLF_2_R_none_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean.pt

done