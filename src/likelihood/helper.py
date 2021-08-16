"""
Setting logger filename
"""
import sys
import logging
from datetime import datetime
import os
from pathlib import Path
import random

identifier = 'stereoset'
# save_path = ('/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/checkpoint/'
#         'Reddit_90M_genderation_LRx8/bias_score/likelihood/')
save_path = ('/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/checkpoint/'
        'Reddit_90M_FT_once_genderation_token/bias_score/likelihood/')
Path(save_path).mkdir(parents=True, exist_ok=True) # create directory if not exist
timestamp = datetime.now().strftime("%m-%d-%H")
random_num = random.randrange(1000)
OUTFILE = save_path + identifier + '_' + timestamp + '_' + str(random_num) + '.log'