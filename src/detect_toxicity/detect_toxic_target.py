#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Output toxicity stats of training data targets
as measured by classifier and string matcher

Sample usage:
```
python detect_toxic_target.py convai2
```

Reference: https://github.com/facebookresearch/ParlAI/blob/2f48dcd9ab7b44b09001807b6a8a5e783972ad59/parlai/scripts/detect_offensive_language.py
"""  
# noqa: E501
import parlai # otherwise module not found in debugger
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
from parlai.utils.misc import TimeLogger
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script
from pprint import pprint
import sys
import os
from tqdm import tqdm
import json
import csv
from collections import defaultdict

MY_DATA='/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data'

class DetectOffensive:
    def __init__(self):
        super().__init__()

        self.stats = {
            'bad_words': [],
            'bad_words_cnt': 0,
            'string_offensive': 0,
            'classifier_offensive': 0,
            'total_offensive': 0,
            'total': 0,
        }
        self.stats2 = self.init_stats2()
        self.offensive_string_matcher = OffensiveStringMatcher()
        self.offensive_classifier = OffensiveLanguageClassifier()


    def report(self):
        """report cumulative stats for all gender tokens"""
        log = {
            'string_offenses%': 100 * (self.stats['string_offensive'] / self.stats['total']),
            'classifier_offenses%': 100
            * (self.stats['classifier_offensive'] / self.stats['total']),
            'total_offenses%': 100 * (self.stats['total_offensive'] / self.stats['total']),
            'word_offenses': self.stats['bad_words_cnt'],
        }
        return log

    def init_stats2(self):
        stats_dict = {
            'f0m0': {
                'string_offensive': 0,
                'classifier_offensive': 0,
                'total_offensive': 0,
                'total': 0,            
                'bad_words_cnt': 0,   
            },
            'f1m0': {
                'string_offensive': 0,
                'classifier_offensive': 0,
                'total_offensive': 0,
                'total': 0,            
                'bad_words_cnt': 0,   
            },
            'f0m1': {
                'string_offensive': 0,
                'classifier_offensive': 0,
                'total_offensive': 0,
                'total': 0,            
                'bad_words_cnt': 0,   
            },
            'f1m1': {
                'string_offensive': 0,
                'classifier_offensive': 0,
                'total_offensive': 0,
                'total': 0,            
                'bad_words_cnt': 0,   
            }
        }

        return stats_dict

    def init_stats3(self):
        # can do stats_dict['f0m0u']['total']+=1 without initialization
        stats_dict = defaultdict(lambda: defaultdict(int))
        return stats_dict

    def report2(self, stat_dict):
        """report stats for each gender token"""
        report = {}
        for token, stats in stat_dict.items():
            if stats['total'] == 0:
                # avoid division by 0 error
                log = {'1.':0,'2.':0,'3.':0,'4.':0,'5.':0,'6.':0,'7.':0,'8.':0}
            else:
                log = {
                    '1. string_offenses%': 100 * (stats['string_offensive'] / stats['total']),
                    '2. classifier_offenses%': 100
                    * (stats['classifier_offensive'] / stats['total']),
                    '3. total_offenses%': 100 * (stats['total_offensive'] / stats['total']),
                    '4. # string offenses: ': stats['string_offensive'],
                    '5. # classifier_offenses': stats['classifier_offensive'],
                    '6. # total offenses': stats['total_offensive'],
                    '7. # total examples': stats['total'],
                    '8. word_offenses_cnt': stats['bad_words_cnt'],
                }
            report[token] = log
        return report

    def classify(self, text):
        offensive = False
        classifier_offensive = False
        self.stats['total'] += 1
        bad_words = self.offensive_string_matcher.contains_offensive_language(text)
        if bad_words:
            self.stats['string_offensive'] += 1
            offensive = True
            self.stats['bad_words'].append(bad_words)
        if text in self.offensive_classifier:
            self.stats['classifier_offensive'] += 1
            offensive = True
            classifier_offensive = True
        if offensive:
            self.stats['total_offensive'] += 1
        return classifier_offensive

    def classify2(self, text, token, task_stats):
        """tag as offensive by string matcher and classifier, 
            update cumulative, token specific stats and task specific stats"""
        offensive = False
        # update total
        # self.stats['total'] += 1
        # self.stats2[token]['total'] += 1
        task_stats[token]['total'] += 1
        bad_words = self.offensive_string_matcher.contains_offensive_language(text)
        if bad_words:
            offensive = True
            # update string_offensive count
            # self.stats['string_offensive'] += 1
            # self.stats2[token]['string_offensive'] += 1
            task_stats[token]['string_offensive'] += 1
            # update bad_words_cnt
            num_bad_words = len(bad_words)
            # self.stats['bad_words_cnt'] += 1
            # self.stats2[token]['bad_words_cnt'] += 1
            task_stats[token]['bad_words_cnt'] += 1
        if text in self.offensive_classifier:
            offensive = True
            # update classifier_offensive count
            # self.stats['classifier_offensive'] += 1
            # self.stats2[token]['classifier_offensive'] += 1
            task_stats[token]['classifier_offensive'] += 1
        if offensive:
            # at least tagged as offensive once
            # self.stats['total_offensive'] += 1
            # self.stats2[token]['total_offensive'] += 1
            task_stats[token]['total_offensive'] += 1
        return task_stats

    def detect(self, path, num_examples=2000):
        i = 0
        for text in self.get_model_responses(path):
            # clear bad_words
            self.stats['bad_words'] = []
            print("---------------------------{}-------------------------------".format(i))
            print(text)
            classifier_offensive = self.classify(text)
            if classifier_offensive:
                print("Classifier detected offensive sentence.")
            if len(self.stats['bad_words']) > 0:
                print(
                    "Offensive words detected: {}".format(', '.join(self.stats['bad_words']))
                )
            self.stats['bad_words_cnt'] += len(self.stats['bad_words'])
            pprint(self.report())
            i+=1
            if i==2000:
                break

    def get_model_responses(self, path):
        """Generator that yields model response from log file"""
        print('Reading {}...'.format(path))
        with open(path, 'r') as f:
            for line in f:
                field_text = line.lstrip().split(' ', 1)
                if len(field_text) != 2:
                    continue
                field = field_text[0]
                text = field_text[1]
                if field == "model:":
                    yield text

    def get_data(self, task, suffix='train'):
        path = os.path.join(MY_DATA, 'genderation_data', task, suffix+'.json')
        print('reading: ', path)
        with open(path, 'r') as f:        
            raw_data = json.load(f)
        return raw_data

    def data_size(self, task, suffix='train'):
        if task == 'stereoset':
            path = os.path.join(MY_DATA, 'stereoset_token', 'stereoset_'+suffix+'_flattened_token.json')
        else:
            path = os.path.join(MY_DATA, 'genderation_data', task, suffix+'.json')
        print('reading: ', path)
        with open(path, 'r') as f:        
            raw_data = json.load(f)
        return len(raw_data)

    def yield_ep(self, task, suffix='train'):
        path = os.path.join(MY_DATA, 'genderation_data', task, suffix+'.json')
        print('reading: ', path)
        with open(path, 'r') as f:        
            raw_data = json.load(f)       
        for ep in raw_data:
            yield ep.get('text', None), ep.get('labels', ep.get('eval_labels', [None]))[0]

    def detect_target_all(self):
        task_list = ['convai2','empathetic_dialogues','wizard_of_wikipedia','blended_skill_talk']
        for task in task_list:
            print('task: ', task)
            task_stats = self.init_stats2()
            raw_data = self.get_data(task)
            for ep in tqdm(raw_data):
                text, target = ep.get('text', None), ep.get('labels', ep.get('eval_labels', [None]))[0]
                token = text[-4:]
                # update task specific stats dict
                task_stats = self.classify2(target, token, task_stats)
            print('task-specific stats:')
            pprint(self.report2(task_stats))
        print('token specific stats across tasks:')
        pprint(self.report2(self.stats2))
        print('cumulative stats across tasks:')
        report = self.report2(self.stats)
        pprint(report)

    def detect_target_task(self, task):
        """new version to export offensive stats by gender token to csv"""
        print('task: ', task)
        
        # raw_data = self.get_data(task)
        # i = 0
        pbar = tqdm(total=self.data_size(task))
        if task != 'stereoset':
            task_stats = self.init_stats2()
            for text, target in self.yield_ep(task):
                token = text[-4:]
                # update task specific stats dict
                task_stats = self.classify2(target, token, task_stats)
                pbar.update(1)
        else:
            task_stats = self.init_stats3()
            for text, target in self.yield_stereoset():
                token = text[-5:]
                # update task specific stats dict
                task_stats = self.classify2(target, token, task_stats)
                pbar.update(1)            
        pbar.close()
        print('task-specific stats:')
        report = self.report2(task_stats)
        pprint(report)
        # export results to csv file
        outfile = os.path.join(MY_DATA, '..', 'dataset_stats', task+'_train_stats.csv')
        with open(outfile, 'w+') as f:
            writer = csv.writer(f)
            for token, token_dict in report.items():
                for key, value in token_dict.items():
                    writer.writerow([token, key, value])
        print('outfile: ', outfile)

    def yield_stereoset(self, suffix='train'):
        path = os.path.join(MY_DATA, 'stereoset_token', 'stereoset_'+suffix+'_flattened_token.json')
        print('reading: ', path)
        with open(path, 'r') as f:        
            raw_data = json.load(f)       
        for ep in raw_data:
            yield ep.get('context', None), ep.get('sentence')           

if __name__ == '__main__':
    task = sys.argv[1]
    DetectOffensive().detect_target_task(task)

    