"""
Count gendered responses and output genderedness stats

Sample usage:
```
python detect_genderedness.py [path to prediction file]
```

Reference: 
https://github.com/facebookresearch/ParlAI/blob/47765c3256a3e91ba4ae50fdc42250ad5f0c0ecf/parlai/tasks/genderation_bias/utils.py
"""

import os
from typing import List, Tuple
import pandas as pd
import sys

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

class Genderedness:
    def __init__(self):
        word_list_folder = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/gender_word_lists'
        self.m_list, self.f_list = self.build_wordlists(word_list_folder)

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

    def output_stats(self, path, verbose=False):
        m_total, f_total, count = 0, 0, 0
        # for model_response in g.get_model_responses(path):
        for model_response in self.get_model_responses(path):
            # m_cnt, f_cnt, formatted = g.count_gender_word(model_response)
            m_cnt, f_cnt, formatted = self.count_gender_word(model_response)
            if verbose:
                print('---------------------------------{}------------------------------------'.format(count))
                print(formatted)
                print('contains male words: {}, contains female words: {}'.format(m_cnt, f_cnt))
                
            m_total += m_cnt
            f_total += f_cnt
            count += 1
            if verbose:
                print('cumulative male %: {:.4f}%, female %: {:.4f}%'.format(m_total/count*100, f_total/count*100))
            if count == 2000:
                break
                
        print('===========================Aggregate Results=================================')
        print('# male responses: {}, # female responses: {}, # responses: {}'.format(m_total, f_total, count))
        print('male %: {:.4f}%, female %: {:.4f}%'.format(m_total/count*100, f_total/count*100))
        return m_total, f_total, count

    def count_gender_word(self, text:str) -> Tuple[int, int, int]:
        """
        Return m_cnt=1 if at least one male word within text, else m_cnt=0 . Similarly for f_cnt.
        :param text:
            text to consider for control token
        :param word_lists:
            tuple of lists for male-specific and female-specific words
        :return token:
            return control token corresponding to input text.
        """
        formatted_text = self.format_text(text)
        m_cnt = 0
        f_cnt = 0

        # text_list = formatted_text.split(' ') # fail to parse more than one space
        text_list = formatted_text.split()

        for word in text_list:
            if word in self.m_list:
                # m_cnt += 1
                m_cnt = 1
            if word in self.f_list:
                # f_cnt += 1
                f_cnt = 1

        return m_cnt, f_cnt, formatted_text

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
            text = text.replace(punc[1], punc[0]) # required to isolate words

        return text

if __name__ == "__main__":
    # pass the path of prediction file as argument
    # first argument 
    path = sys.argv[1]
    print('path: ', path)
    
    g = Genderedness()
    g.output_stats(path, verbose=True)

