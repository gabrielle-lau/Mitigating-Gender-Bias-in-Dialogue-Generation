#!/bin/bash
#SBATCH -J GBNLGt
#SBATCH -A MLMI-myl40-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p pascal
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! ############################################################
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment
module load cuda/10.2 intel/mkl/2017.4

MODEL=Reddit_90M_FT_once_genderation_token
BDIR=/rds/project/wjb31/rds-wjb31-nmt2020/myl40
MLMI8DIR=/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project
CHECKPOINT=$MLMI8DIR/checkpoint/$MODEL
MODEL_FOLDER=$MLMI8DIR/my_data/models/tutorial_transformer_generator/
BS=32 #16
LR=8e-06 #1e-06

JOBID=$SLURM_JOB_ID
mkdir -p $CHECKPOINT/
LOG=$BDIR/logs/$JOBID.$MODEL.log

echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG

# finetune Reddit 90M with 5 datasets labelled with GBS-tokens
python ParlAI/parlai/scripts/train_model.py \
-t blended_skill_talk_token,wizard_of_wikipedia_token,convai2_token,empathetic_dialogues_token,stereoset_token \
--multitask-weights 2,2,2,2,2 \
--datapath $MLMI8DIR/my_data \
--init-model zoo:tutorial_transformer_generator/model \
--dict-file zoo:tutorial_transformer_generator/model.dict \
-m transformer/generator \
--embedding-size 512 \
--n-layers 8 \
--ffn-size 2048 \
--dropout 0.1 \
--n-heads 16 \
--learn-positional-embeddings True \
--n-positions 512 \
--variant xlm \
--activation gelu \
--fp16 True \
--text-truncate 512 \
--label-truncate 128 \
--dict-tokenizer bpe \
--dict-lower True \
-lr $LR \
--optimizer adamax \
--lr-scheduler reduceonplateau \
--gradient-clip 0.1 \
--betas 0.9,0.999 \
--update-freq 1 \
--attention-dropout 0.0 \
--relu-dropout 0.0 \
-vp 15 \
-stim 60 \
-vme 20000 \
-bs $BS \
-vmt ppl \
-vmm min \
-veps 0.25 \
--model-file $CHECKPOINT/$MODEL.1 \
-wblog True \
--wandb-project gender_bias \
--wandb-entity laumingyi \
--skip-generation False \
--model-parallel True \
--save-after-valid False \
--save-every-n-secs 1800 \
--log-every-n-steps 10 \
&>> $LOG

# --wandb-id "2b31yp5t" \
# --wandb-resume "must" \

echo "Time: `date`" >> $LOG
