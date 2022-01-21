#!/bin/bash


source activate snn_gpu
module load cuda
echo `which python`
cd $PRODUCTS/classifiers/supernnova
echo "#################TIMING  Starting here:   `date`"

#J17
python run.py --data --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --redshift_label REDSHIFT_FINAL --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_J17 --raw_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/1_SIM/14XDES_J17/PIP_AM_DES5YR_SIMS_14XDES_J17  
for seed in 0 100 1000 55 30469
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_J17 --cyclic --model vanilla  --redshift zspe --norm cosmo  --train_rnn --seed $seed
done

#NOPEC
python run.py --data --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --redshift_label REDSHIFT_FINAL --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_NOPEC --raw_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/1_SIM/14XDES_NOPEC/PIP_AM_DES5YR_SIMS_14XDES_NOPEC  
for seed in 0 100 1000 55 30469
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_NOPEC --cyclic --model vanilla  --redshift zspe --norm cosmo  --train_rnn --seed $seed
done

#OLDIA
python run.py --data --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --redshift_label REDSHIFT_FINAL --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_OLDIA --raw_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/1_SIM/14XDES_OLDIA/PIP_AM_DES5YR_SIMS_14XDES_OLDIA 
for seed in 0 100 1000 55 30469
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_OLDIA --cyclic --model vanilla  --redshift zspe --norm cosmo  --train_rnn --seed $seed
done

#PSNID
python run.py --data --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --redshift_label REDSHIFT_FINAL --phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_PSNID --raw_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/1_SIM/14XDES_PSNID/PIP_AM_DES5YR_SIMS_14XDES_PSNID  
for seed in 0 100 1000 55 30469
do
  echo $seed
  python run.py --use_cuda --sntypes '{"1": "Ia", "11": "II", "20": "II", "91": "II", "101": "Ia", "111": "II", "120": "II", "191": "II"}' --batch_size 128 --dump_dir $PIPPIN_OUTPUT/AM_DES5YR_SIMS/snndump_14XDES_PSNID --cyclic --model vanilla  --redshift zspe --norm cosmo  --train_rnn --seed $seed
done


