#!/bin/bash

#Set permission by
#chmod +x script-name-here.sh
#script-name-here.sh

MLMI8DIR=/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project
SWAG_DIR=$MLMI8DIR/my_data/swagaf/data

python $MLMI8DIR/src/roberta/run_multiple_choice.py \
--model_type roberta \
--task_name swag \
--model_name_or_path roberta-base \
--do_train \
--do_eval \
--do_lower_case \
--data_dir $SWAG_DIR \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--output_dir $MLMI8DIR/checkpoint/Roberta_MC_trial/ \
--per_gpu_eval_batch_size=16 \
--per_gpu_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output \
--evaluate_during_training