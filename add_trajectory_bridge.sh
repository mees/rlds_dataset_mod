: '
Script for downloading, cleaning and resizing Open X-Embodiment Dataset (https://robotics-transformer-x.github.io/)

Performs the preprocessing steps:
  1. Downloads mixture of 25 Open X-Embodiment datasets
  2. Runs resize function to convert all datasets to 256x256 (if image resolution is larger) and jpeg encoding
  3. Fixes channel flip errors in a few datsets, filters success-only for QT-Opt ("kuka") data

To reduce disk memory usage during conversion, we download the datasets 1-by-1, convert them
and then delete the original.
We specify the number of parallel workers below -- the more parallel workers, the faster data conversion will run.
Adjust workers to fit the available memory of your machine, the more workers + episodes in memory, the faster.
The default values are tested with a server with ~120GB of RAM and 24 cores.
'

CONVERSION_DIR=/scratch/partial_datasets/oiermees/bridge_trajectory
N_WORKERS=20                  # number of workers used for parallel conversion --> adjust based on available RAM
MAX_EPISODES_IN_MEMORY=200    # number of episodes converted in parallel --> adjust based on available RAM

# increase limit on number of files opened in parallel to 20k --> conversion opens up to 1k temporary files
# in /tmp to store dataset during conversion
ulimit -n 20000
python3 modify_rlds_dataset.py --dataset=bridge_orig --data_dir=/shared/karl/data/ --target_dir=$CONVERSION_DIR --mods=visual_trajectory --n_workers=$N_WORKERS --max_episodes_in_memory=$MAX_EPISODES_IN_MEMORY

