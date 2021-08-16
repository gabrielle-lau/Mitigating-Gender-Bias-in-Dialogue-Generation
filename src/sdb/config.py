#!/usr/bin/env python

# identifier='hostile_sdb'
identifier='hostile_prompt'

model='Blender_90M_orig_zoo'
# model='Reddit_90M_genderation_LRx8'
# model='Reddit_90M_FT_once_genderation_token'

compare_mode=True
skip_save=True # set to True if compare_mode True but don't want to save tensor
fixed_scale_mode=False
floored_scale_mode=False
compare_with='hostile_sdb'
decay_const=100
fixed_scale_const=0.01
save_path=('/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/checkpoint/'
                +model+'/sdb/')
notes=''