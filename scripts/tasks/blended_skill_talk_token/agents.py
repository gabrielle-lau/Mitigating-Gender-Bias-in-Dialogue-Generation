'''
agents.py for blended_skill_talk dataset
'''

from parlai.core.teachers import (
    DialogTeacher, 
    create_task_agent_from_taskname
)
from parlai.utils.io import PathManager
from .build import build
import json
import os

class BSTTeacher(DialogTeacher):
    """
    Genderation
    """
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        # whatever is placed into datafile will be passed as the argument to
        # setup_data in the next section.
        opt['datafile'] = os.path.join(opt['datapath'], 'genderation_data_token', 'blended_skill_talk', suffix + '.json')
        # matches id in task_list.py
        self.id = 'blended_skill_talk_token'
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            raw_data = json.load(data_file)
        for ep in raw_data:
            # ep is a dict of an episode of data
            yield {"text": ep['text'], "labels": ep.get('labels', ep.get('eval_labels', None))}, ep['episode_done']


class DefaultTeacher(BSTTeacher):
    pass

def create_agents(opt):
    if not opt.get('interactive_task', False):
        return create_task_agent_from_taskname(opt)
    else:
        # interactive task has no task agents (they are attached as user agents)
        return []