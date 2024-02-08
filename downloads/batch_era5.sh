#!/bin/bash
#SBATCH --job-name=diego_era       # job's name
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH -N 1                        # number of nodes (or --nodes=1)
#SBATCH -n 8                        # number of tasks (or --tasks=8)
#SBATCH --mem-per-cpu=8G            # memory per core
#SBATCH --time=24:00:00             # Wall Time 24h
#SBATCH --account=helpdesk_swotlr   # MANDATORY : account  ( launch myaccounts to list your accounts)

# not sure line below is working
##SBATCH --export=none               # To start the job with a clean environnement and source of ~/.bashrc

date

# to go to the submit directory
cd ${SLURM_SUBMIT_DIR}

# Add here the name of the script or executable to run
#EXE="python batch.py"
EXE="python batch_era5.py"

source ~/.bashrc


# run first:
#   conda activate /work/HELPDESK_SWOTLR/commun/envs/py311_dev/

#conda activate py311_dev
#conda activate /work/HELPDESK_SWOTLR/commun/envs/py311_dev/ 
#conda list

# launch the script/executable on the batch node(s)
$EXE > job_$SLURM_JOB_ID.log

# for L2 downloads
#ls -lth /work/HELPDESK_SWOTLR/commun/data/swot/cache/
# ERA5
ls -lth /work/HELPDESK_SWOTLR/swot_diego/era5

echo "All done (.sh)"

date

