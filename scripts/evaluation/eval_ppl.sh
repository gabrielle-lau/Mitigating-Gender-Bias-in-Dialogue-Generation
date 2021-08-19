#!/bin/bash
#SBATCH -J GBDGe
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
CMD=eval

# # ## declare an array variable
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

# declare -a arr=(
# "f0m0"
# "f1m0"
# "f0m1"
# "f1m1")

bin=${arr[$SLURM_ARRAY_TASK_ID]}
DEST=$CHECKPOINT/eval_model/stereoset # ADD FOLDER FOR STEREOSET
mkdir -p $DEST
LOG=$DEST/$CMD.$bin.$JOBID.log

echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG

echo "$bin" &>> $LOG

CMD="parlai eval_model --task genderation_bias:controllable_task:stereoset_unique \
-dp $MLMI8DIR/my_data --metrics ppl --fixed_control $bin --dynamic-batching full \
--model-file $CHECKPOINT/Reddit_90M_FT_once_genderation_token.1.checkpoint \
-dt valid --num_examples 425 --report-filename $LOG.report"

echo $CMD >> $LOG

$CMD &>> $LOG

echo "Time: `date`" >> $LOG
