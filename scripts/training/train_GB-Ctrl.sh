#!/bin/bash
#SBATCH -J GBNLGt
#SBATCH -A MLMI-myl40-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p pascal
#! ############################################################
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment
module load cuda/10.2 intel/mkl/2017.4

MODEL=Reddit_90M_genderation_LRx8
BDIR=/rds/project/wjb31/rds-wjb31-nmt2020/myl40
MLMI8DIR=/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project
CHECKPOINT=$MLMI8DIR/checkpoint/$MODEL
MODEL_FOLDER=$MLMI8DIR/my_data/models/tutorial_transformer_generator/
BS=32 #16
LR=8e-6 #1e-6 
JOBID=$SLURM_JOB_ID
mkdir -p $CHECKPOINT/
LOG=$CHECKPOINT/train_log/$JOBID.$MODEL.zoo_to_1.log

echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG

# finetune Reddit 90M with 4 datasets labelled with GB-tokens
parlai train_model \
-t blended_skill_talk_genderation,wizard_of_wikipedia_genderation,convai2_genderation,empathetic_dialogues_genderation \
--datapath $MLMI8DIR/my_data \
--multitask-weights 1,3,3,3 \
--init-model $CHECKPOINT/$MODEL.1 \
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
--wandb-id "3k1fuh14" \
--wandb-resume "must" \
&>> $LOG



echo "Time: `date`" >> $LOG
