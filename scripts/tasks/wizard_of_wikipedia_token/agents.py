'''
agents.py for wizard_of_wikipedia dataset
'''

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from .build import build

import json
import os


class WoWTeacher(DialogTeacher):
    """
    Genderation
    """
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        # whatever is placed into datafile will be passed as the argument to
        # setup_data in the next section.
        opt['datafile'] = os.path.join(opt['datapath'], 'genderation_data_token', 'wizard_of_wikipedia', suffix + '.json')
        # matches id in task_list.py
        self.id = 'wizard_of_wikipedia_token'
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            raw_data = json.load(data_file)
        for ep in raw_data:
            # ep is a dict
           yield {"text": ep['text'], "labels": ep.get('labels', ep.get('eval_labels', None))}, ep['episode_done']
                
class DefaultTeacher(WoWTeacher):
    pass