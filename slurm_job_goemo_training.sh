#!/bin/bash

# https://www.sc.uni-leipzig.de/05_Instructions/Slurm/
# https://hpc.nmsu.edu/onboarding/supercomputing/slurm/workflow/

## This is just a template
## double ## will disable a line completely
## single #SBATCH marks active lines

##SBATCH --job-name myJobName 	    ## The name that will show up in the queue
##SBATCH --output myJobName-%j.out   ## Filename of the output; default is slurm-[joblD].out
##SBATCH --partition normal          ## The partition to run in; default = normal
##SBATCH --nodes 1 		    ## Number of nodes to use; default = 1
##SBATCH --ntasks 3 		    ## Number of tasks (analyses) to run; default = 1
##SBATCH --cpus-per-task 16 	    ## The num of threads the code will use; default = 1
##SBATCH --mem-per-cpu 700M          ## Memory per allocated CPU
##SBATCH --time 0-00:10:00	    ## Time for analysis (day-hour:min:sec)
##SBATCH --mail-user yourlD@nmsu.edu ## Your email address
##SBATCH --mail-type BEGIN 	    ## Slurm will email you when your job starts
##SBATCH --mail-type END 	    ## Slurm will email you when your job ends
##SBATCH --mail-type FAIL            ## Slurm will email you when your job fails
##SBATCH --get-user-env 	## Passes along environmental settings

##########################################################################################

# Name for the job
# Allows you to specify a custom string to identify your job in the queue
#SBATCH --job-name=Goemotions-Training

# Output and Error Files
#SBATCH --output=jobfiles/log/%x.out-%j
#SBATCH --error=jobfiles/log/%x.error-%j


# Partition to run the jobs on
# sirius	2 days	1 node	big memory applications
# sirius-long	10 days	1 node	big memory applications
#--------------------------------------------------------
# polaris	2 days	10 nodes	general computation
# polaris-long	42 days	4 nodes	long runtime
#--------------------------------------------------------
# clara		2 days	29 nodes w/ GPUs	general and GPU computation
# clara-long	10 days	6 nodes w/ GPUs	long running GPU jobs
#--------------------------------------------------------
# paula		2 days	12 nodes w/ GPUs	general and GPU computation
# paul		2 days	32 nodes	general computation
#--------------------------------------------------------
#SBATCH --partition=clara

#SBATCH --nodes 1

# Sets the number of tasks (i.e. processes)
#SBATCH --ntasks=1

#SBATCH --cpus-per-task 4	    ## The num of threads the code will use; default = 1

# Request real memory required per node.
# Default unit is megabytes.
# The options --mem,                memory required per node
#             --mem-per-cpu, and    memory required per task (goes times n-tasks)
#             --mem-per-gpu         memory per GPU task (i.e. process), comprises memory by CPU and GPU.
# are mutually exclusive.
# Requesting 1024MB per tasks with 8 GPU tasks amounts to 8GB in total for the job.
# The total includes the memory for the CPU part as well as the memory used on the GPU!
# on a V100 => 40 GB
#SBATCH --mem=32G

# Number and type of GPUs
# Requesting 1 GeForce RTX 2080Ti
#SBATCH --gres=gpu:rtx2080ti:1
# Requesting 1 Tesla V100
## SBATCH --gres=gpu:v100:1
# Requesting 4 Tesla A30
## SBATCH --gres=gpu:a30:4

# The maximal runtime (Walltime) for your job
# Format: [hours:]minutes[:seconds], e.g. "30" equals 30 min, "3:20" equals 3h and 20 min
# Alternate format: days-hours[:minutes][:seconds], e.g. "2-0" equals 2 days, "1-5:20" equals 1 day 5h 20 min
#SBATCH --time=2-0

#SBATCH --mail-user ralf.koenig@studserv.uni-leipzig.de ## Your email address
#SBATCH --mail-type BEGIN 	    ## Slurm will email you when your job starts
#SBATCH --mail-type END 	    ## Slurm will email you when your job ends
#SBATCH --mail-type FAIL            ## Slurm will email you when your job fails

# das hier gibt es wirklich
module load Python/3.10.8-GCCcore-12.2.0-bare
module load JupyterLab/3.1.6-GCCcore-11.2.0

jupyter nbconvert --to notebook --execute notebooks/bert_models.ipynb --output tf-job-%j.ipynb
