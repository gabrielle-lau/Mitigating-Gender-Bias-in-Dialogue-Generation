"""
Append correct GBS-token to context of each StereoSet example
Source: Processed (flattened) data in my_data/stereoset/
Dest: Appended token data in my_data/stereoset_token/

Reference: https://github.com/facebookresearch/ParlAI/blob/47765c3256a3e91ba4ae50fdc42250ad5f0c0ecf/parlai/tasks/genderation_bias/utils.py
"""

import os
from typing import List, Tuple
import pandas as pd
import json

PUNCTUATION_LST = [
    (' .', '.'),
    (' !', '!'),
    (' ?', '?'),
    (' ,', ','),
    (" ' ", "'"),
    (" . . . ", "... "),
    (" ( ", " ("),
    (" ) ", ") "),
    (" ; ", "; "),
]

TOKENS=[ 
'f0m0s',
'f1m0s',
'f0m1s',
'f1m1s',
'f0m0a',
'f1m0a',
'f0m1a',
'f1m1a',
'f0m0u',
'f1m0u',
'f0m1u',
'f1m1u'
]

STEREO=['a', 's', 'u']

class StereoToken:
    def __init__(self):
        word_list_folder = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/gender_word_lists'
        self.m_list, self.f_list = self.build_wordlists(word_list_folder)
        self.dest = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/stereoset_token_incorrect/'

    def load_data(self, path):
        """"load stereoset intersentence data from path & return list"""
        with open(path,'r') as f:
            all_data=json.load(f)
        return all_data

    def append_token(self, path, name):
        """Append 1 of 12 tokens to original dataset at path"""
        data = self.load_data(path)
    
        for ex in data:
            context = ex['context']
            gold_label = ex['gold_label']
            sentence = ex['sentence']
            
            formatted_text = self.format_text(sentence)
            gender_bin = self.get_gender_bin(formatted_text)
            token = gender_bin + gold_label[0]
            # modify in place
            ex['context'] = context + ' ' + token
            # print(ex['context'])
        
        with open(self.dest+name+'.json', 'w+') as f:
            json.dump(data, f, indent = 4)
        
        print('Successfully saved: ', self.dest+name+'.json')

    def get_gender_bin(self, text:str) -> Tuple[int, int, int]:
        """
        Return m_cnt=1 if at least one male word within text, else m_cnt=0 . Similarly for f_cnt.
        :param text:
            text to consider for control token
        :param word_lists:
            tuple of lists for male-specific and female-specific words
        :return token:
            return control token corresponding to input text.
        """
        m_cnt = 0
        f_cnt = 0

        text_list = text.split()

        for word in text_list:
            if word in self.m_list:
                m_cnt = 1
            if word in self.f_list:
                f_cnt = 1

        if f_cnt == 0 and m_cnt == 0:
            return 'f0m0'
        elif f_cnt == 0 and m_cnt > 0:
            return 'f0m1'
        elif f_cnt > 0 and m_cnt == 0:
            return 'f1m0'
        else:
            return 'f1m1'

    def build_wordlists(self, folder: str) -> Tuple[List[str], List[str]]:
        """
        Load list of explicitly gendered words.
        Words taken from <https://github.com/uclanlp/gn_glove/blob/master/wordlist/>.
        Examples include brother, girl, actress, husbands, etc.
        """
        male_words = os.path.join(folder, 'male_word_file.txt')
        female_words = os.path.join(folder, 'female_word_file.txt')

        with open(male_words, 'r') as f:
            male = f.read().splitlines()

        with open(female_words, 'r') as f:
            female = f.read().splitlines()

        return male, female

    def format_text(self, text: str, lower: bool = True) -> str:
        """
        Space punctuation and lowercase text.
        :param text:
            text to lowercase
        :param lower:
            whether to lowercase or not
        :return text:
            return formatted text.
        """
        if lower:
            text = text.lower()
        for punc in PUNCTUATION_LST:
            text = text.replace(punc[1], punc[0])

        return text


if __name__ == '__main__':
    # processed, flattened stereoset data
    path = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/stereoset/stereoset_train_flattened.json'
    # path = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/stereoset/stereoset_test_flattened.json'
    
    s = StereoToken()
    filename = path.rsplit('/', 1)[1] # split from right once
    filename = filename.split('.')[0]
    print('filename: ', filename)
    s.append_token(path, name=filename+'_token')

    