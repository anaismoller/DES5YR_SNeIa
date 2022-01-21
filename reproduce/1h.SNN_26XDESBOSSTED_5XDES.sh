#!/bin/bash

source activate snn_gpu
module load cuda
echo `which python`
# Beware, batch size of database may need reduction
cd $PRODUCTS/classifiers/supernnova

echo "#################TIMING  Starting here:   `date`"

# database
python run.py --data --data_testing --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --redshift_label REDSHIFT_FINAL --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS_TEST/1_SIM/5XDES/snndump_5XDES --raw_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS_TEST/1_SIM/5XDES/PIP_AM_DES5YR_SIMS_TEST_5XDES/  --fits_dir $PIPPIN_OUTPUT/AM_DATA5YR/2_LCFIT/D_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz

# with redshift
for seed in 0 100 1000 55 30469 30496 49 7510 1444 9 1 2 303 4444 537
do
  echo $seed
  python run.py --use_cuda --redshift zspe --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 1024 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS_TEST/1_SIM/5XDES/snndump_5XDES --model vanilla --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_26XBOOSTEDDES/models/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_1024_True_mean.pt
done

echo "finished"