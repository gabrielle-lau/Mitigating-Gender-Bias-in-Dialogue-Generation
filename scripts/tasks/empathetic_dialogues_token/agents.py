'''
agents.py for empathetic_dialogue dataset
'''

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from .build import build

import json
import os


class EmpatheticDialoguesTeacher(DialogTeacher):
    """
    Genderation
    """
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        # whatever is placed into datafile will be passed as the argument to
        # setup_data in the next section.
        opt['datafile'] = os.path.join(opt['datapath'], 'genderation_data_token', 'empathetic_dialogues', suffix + '.json')
        # matches id in task_list.py and task folder name
        self.id = 'empathetic_dialogues_token'
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            raw_data = json.load(data_file)
        for ep in raw_data:
            # ep is a dict
           yield {"text": ep['text'], "labels": ep.get('labels', ep.get('eval_labels', None))}, ep['episode_done']
                
class DefaultTeacher(EmpatheticDialoguesTeacher):
    pass