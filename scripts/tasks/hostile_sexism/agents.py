'''
agents.py for wizard_of_wikipedia dataset
'''

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from .build import build

import json
import os


class SexismTeacher(DialogTeacher):
    """
    Genderation
    """
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        # whatever is placed into datafile will be passed as the argument to
        # setup_data in the next section.
        opt['datafile'] = os.path.join(opt['datapath'], 'ambivalent_sexism', 
                            'hostile_sexism', 'hostile_prompt.json')
        # matches id in task_list.py
        self.id = 'hostile_sexism'
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            raw_data = json.load(data_file)
        suffix = " . acceptable ?"
        for ep in raw_data['data']:
            # ep is a dict
            yield {"text": str(ep['tweet'])+suffix}, True
            yield {"text": "why ?"}, False    
class DefaultTeacher(SexismTeacher):
    pass