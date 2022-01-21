#!/bin/bash


source activate snn_gpu
module load cuda
echo `which python`
cd $PRODUCTS/classifiers/supernnova


echo "#################TIMING  Starting here:   `date`"
# if scripts are ran in sequence, database reconstruction is not needed
# python run.py --data --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --redshift_label REDSHIFT_FINAL --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --raw_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/1_SIM/14XDES/PIP_AM_DES5YR_SIMS_14XDES

for batch in 128 512 1024
do
   echo "batch $batch"
   python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size $batch --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --cyclic --model vanilla   --redshift zspe --norm cosmo  --train_rnn --data_fraction 0.2
done

for dropout in 0.1 0.2
do
   echo "dropout $dropout"
   python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --cyclic --model vanilla   --redshift zspe --norm cosmo  --train_rnn --data_fraction 0.2 --dropout $dropout
done

echo "no cyclic"
python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --model vanilla   --redshift zspe --norm cosmo  --train_rnn --data_fraction 0.2

echo "no bidirectional"
python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --cyclic --model vanilla   --redshift zspe --norm cosmo  --train_rnn --data_fraction 0.2 --bidirectional False

for hidden in 64 128
do
   echo "hidden dim $hidden"
   python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --cyclic --model vanilla   --redshift zspe --norm cosmo  --train_rnn --data_fraction 0.2 --hidden_dim $hidden
done

for layers in 3 4
do
   echo "layers $layers no cyclic"
   python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --model vanilla   --redshift zspe --norm cosmo  --train_rnn --data_fraction 0.2 --num_layers $layers
   python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --model vanilla   --redshift zspe --norm cosmo  --train_rnn --data_fraction 0.2 --num_layers $layers --hidden_dim 64
done

python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --cyclic --model vanilla   --redshift zspe --norm cosmo  --train_rnn --data_fraction 0.2 --num_layers 4 --hidden_dim 64

# echo "cyclic_phases 20 40 60"
python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES --cyclic --model vanilla   --redshift zspe --norm cosmo  --train_rnn --data_fraction 0.2 --cyclic_phases 20 40 60 --hidden_dim 63

