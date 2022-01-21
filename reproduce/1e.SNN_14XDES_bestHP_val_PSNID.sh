#!/bin/bash


source activate snn_gpu
module load cuda
echo `which python`
# $PRODUCTS is where SuperNNova is installed
cd $PRODUCTS/classifiers/supernnova

echo "#################TIMING  Starting here:   `date`"

# with redshift
for seed in 0 100 1000 55 30469
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_PSNID --model vanilla  --redshift zspe --norm cosmo  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES/models/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_512_True_mean/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_512_True_mean.pt
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_PSNID --model vanilla  --redshift zspe --norm cosmo_quantile  --validate_rnn --seed $seed --num_layers 4 --hidden_dim 64 --model_files $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES/models/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_512_True_mean/vanilla_S_${seed}_CLF_2_R_zspe_photometry_DF_1.0_N_cosmo_quantile_lstm_64x4_0.05_512_True_mean.pt
done
