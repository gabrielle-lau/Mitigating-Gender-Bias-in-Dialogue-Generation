'''
agents.py for Convai2 dataset
'''

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from .build import build

import json
import os


class Convai2Teacher(DialogTeacher):
    """
    Genderation
    """
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        # whatever is placed into datafile will be passed as the argument to
        # setup_data in the next section.
        opt['datafile'] = os.path.join(opt['datapath'], 'genderation_data', 'convai2', suffix + '.json')
        # matches id in task_list.py
        self.id = 'convai2_genderation'
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            raw_data = json.load(data_file)
        for ep in raw_data:
            # ep is a dict
            yield {"text": ep['text'], "labels": ep.get('labels', ep.get('eval_labels', None))}, ep['episode_done']
               
class DefaultTeacher(Convai2Teacher):
    pass