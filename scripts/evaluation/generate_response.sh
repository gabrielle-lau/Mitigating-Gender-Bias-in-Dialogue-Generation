#!/bin/bash
#SBATCH -J GBDGp
#SBATCH -A MLMI-myl40-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH -p skylake,cclake
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-11
#! ############################################################
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded

BDIR=/rds/project/wjb31/rds-wjb31-nmt2020/myl40
MLMI8DIR=/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project
CHECKPOINT=$MLMI8DIR/checkpoint/GB-Ctrl

JOBID=$SLURM_JOB_ID
CMD=pred

## declare an array variable
declare -a arr=(
"f0m0s"
"f0m0a"
"f0m0u"
"f1m0s"
"f1m0a"
"f1m0u"
"f0m1s"
"f0m1a"
"f0m1u"
"f1m1s"
"f1m1a"
"f1m1u")

## CHANGE SBATCH array to match arr size!!!
# declare -a arr=(
# "f0m0"
# "f1m0"
# "f0m1"
# "f1m1")

bin=${arr[$SLURM_ARRAY_TASK_ID]}
DEST=$CHECKPOINT/model_response
mkdir -p $DEST
LOG=$DEST/$CMD.$bin

echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG

echo "$bin" &>> $LOG

# generate system responses
CMD="parlai display_model --task genderation_bias:controllable_task:convai2 \
-dt valid --fixed_control $bin --datapath $MLMI8DIR/my_data \
--model-file $CHECKPOINT/GB-Ctrl.checkpoint \
--skip-generation False --num_examples 2000"

echo $CMD >> $LOG

$CMD &>> $LOG

echo "Time: `date`" >> $LOG
