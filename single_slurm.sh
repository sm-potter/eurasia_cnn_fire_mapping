#!/bin/sh
##SBATCH --export=ALL
#SBATCH -N1
##SBATCH --nodelist=gpu012
##SBATCH --ntasks=1
#SBATCH --qos=long
#SBATCH --time=5-0
#SBATCH --cpus-per-task=1
#SBATCH -G4
##SBATCH --mem-per-cpu=400G

# Rscript /home/spotter5/cnn_mapping/v4/burned_area_calculations_landsat.r $1 
#Rscript /home/spotter5/cnn_mapping/v4/burned_area_calculations_landsat_om_com.r $1 
# Rscript /home/spotter5/cnn_mapping/v4/fire_cci_raw_extract.r $1 
# python /home/spotter5/cnn_mapping/v5/train_cnn_dNBR.py 
# python /home/spotter5/cnn_mapping/v5/train_cnn_VI.py 
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_test_gpu.py 
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia.py 
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_combined.py 
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_combined_full_fire.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_combined_full_fire_old_dnbr.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_combined_full_fire_old_dnbr_65_15_20.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_combined_full_fire_65_15_20.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_combined_full_fire_85.py:q!
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_full_fire.py
#python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_full_fires_old_dnbr.py
#python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_full_fires_65_15_20.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_full_fires_old_dnbr_65_15_20.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_full_fires_65_15_20.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_full_fires_85.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_2015_2019.py 
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_2015_2019.py 
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_combined_2015_2019.py 
# python /home/spotter5/cnn_mapping/Russia/IOU_test_grid_ecoregions.py
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_full_fires_old_dnbr_cv.py
python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_full_fires_cv.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_combined_full_fire_old_dnbr_cv.py
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_combined_full_fire_dnbr_cv.py

##export LD_LIBRARY_PATH=/home/spotter5/.conda/envs/deeplearning3/lib:$LD_LIBRARY_PATH
##export XLA_FLAGS='--xla_gpu_cuda_data_dir=/home/spotter5/.conda/envs/deeplearning3/lib'
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
##export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/spotter5/.conda/envs/deeplearning/lib/python3.8/site-packages/jaxlib/cuda/nvvm"
# python /home/spotter5/cnn_mapping/Russia/train_cnn_VI_russia_full_fires_cv.py











