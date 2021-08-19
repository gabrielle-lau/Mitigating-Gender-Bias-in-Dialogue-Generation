"""
Compute modelling error for generating confusion matrices,
using files containing target likelihoods given context and token.
"""
import os
from datetime import datetime
import sys
import json
from collections import Counter
import csv
import numpy as np
import random

token2int = {
    'f0m0a':0,
    'f0m0s':1,
    'f0m0u':2,
    'f0m1a':3,
    'f0m1s':4,
    'f0m1u':5,
    'f1m0a':6,
    'f1m0s':7,
    'f1m0u':8,
    'f1m1a':9,
    'f1m1s':10,
    'f1m1u':11
}

gender2int = {
    'f0m0': 0,
    'f0m1': 1,
    'f1m0': 2,
    'f1m1': 3
}

class ModelError:
    def __init__(self, correct, incorrect, dest):
        super().__init__()
        # init all the paths
        self.correct = correct
        self.incorrect = incorrect
        self.dest = dest

    def compare(self):
        with open(self.correct, 'r') as f:
            correct = f.readlines()
        with open(self.incorrect, 'r') as f:
            incorrect = f.readlines()    

        total = len(correct)
        error_cnt = 0
        if len(correct) != len(incorrect):
            print('Error: number of rows in two files not match')
            return

        timestamp = datetime.now().strftime("%m-%d-%H")
        correction_log = os.path.join(dest, 'correction.csv')
        with open(correction_log, 'w+') as f:
            for i in range(total):
                # error if posterior of target given incorrect token > given correct token
                if correct[i] < incorrect[i]:
                    error_cnt += 1
                    f.write('incorrect\n')
                else:
                    f.write('correct\n')
        print('Results saved to: ', correction_log)

        log = os.path.join(dest, 'model_error_'+timestamp)
        with open(log, 'w+') as f:
            f.write('correct: {} \n'.format(self.correct))
            f.write('incorrect: {} \n'.format(self.incorrect))
            f.write('# errors: {} \n'.format(error_cnt))
            f.write('# examples: {} \n'.format(total))
            f.write('error%: {:.4f} \n'.format(error_cnt/total*100))
        
        print('Results saved to: ', log)

    def two_tokens(self, path_correct, path_incorrect, tok_len=4, total=2000):
        if path_incorrect:
            with open(path_incorrect,'r') as f:
                data_incorrect = json.load(f)
        else:
            # f0m0u always
            incorrect_token = 'f0m0u'

        i = 0
        if tok_len == 4:
            stereo='u'
        else:
            stereo=''

        with open(path_correct,'r') as f:
            data_correct = json.load(f)

        log = os.path.join(dest, 'true_false_tokens.csv')
        with open(log, 'w+') as f:  
            while True:      
                context = data_correct[i].get('text', data_correct[i].get('context'))
                correct_token = context[-tok_len:]

                if path_incorrect:
                    context = data_incorrect[i].get('text', data_correct[i].get('context'))
                    incorrect_token = context[-tok_len:]
                    f.write(correct_token+stereo+','+incorrect_token+stereo+'\n')
                else:
                    f.write(correct_token+stereo+','+incorrect_token+'\n')
                i += 1
                if i == total:
                    print('results saved to: ', log)
                    return

    def true_pred(self, correction_path, true_false_path):
        c = Counter()
        with open(correction_path, 'r') as f:
            # correction = f.readlines() 
            # to remove newline character
            csvread = csv.reader(f)
            correction = list(csvread)
        with open(true_false_path, 'r') as f:
            # tf = f.readlines()
            csvread = csv.reader(f)
            tf = list(csvread)
        log = os.path.join(dest, 'true_pred_tokens.csv')
        total = len(correction)
        with open(log, 'w+') as f:  
            for i in range(total):
                correct, incorrect = tf[i][0], tf[i][1]
                pred = correct if correction[i][0]=='correct' else incorrect
                f.write(correct+','+pred+'\n')
                c[correct+', '+pred]+=1
        print('results saved to: ', log)

        counter_log = os.path.join(dest, 'counter.csv')
        with open(counter_log, 'w+') as f:
            for key, value in c.items():
                f.write('{}, {}\n'.format(key, value))
        print('results saved to: ', counter_log)

    def max_likelihood(self, dirname, sorted_files, stereoset=False):
        # create array to store likelihoods from each file as a row
        if stereoset:
            arr = np.empty((0,1275),float)
        else:
            arr = np.empty((0,2000),float)
        i = 0
        # int2token = {}
        for filename in sorted_files:
            print('{}: {}'.format(i, filename))
            # int2token[str(i)]=filename
            path = os.path.join(dirname,filename)
            with open(path, 'r') as f:
                # avoid newline character in likelihood list
                likelihood = f.read().splitlines()
                arr = np.vstack((arr,np.array(likelihood)))
                i+=1
        # MLE
        max_index = np.argmax(arr, axis=0)
        
        folder, _ = dirname.rsplit('/',1)
        dest = os.path.join(folder, 'confusion_temp')
        random_num = str(random.randint(100,900))
        pred_file = os.path.join(dest,'pred_indices-'+random_num+'.csv')

        #save indices array
        np.savetxt(pred_file, max_index.astype(int), fmt='%i', delimiter = ",")
        
        print('saved results to: ', pred_file)

        log = os.path.join(dest, 'max_likelihood-'+random_num+'.log')
        with open(log, "w") as f:
            f.write('directory of source files: {}\n'.format(dirname))
            for i, filename in enumerate(sorted_files):
                f.write('{}: {}\n'.format(i, filename))
        
        print('saved results to: ', log)

    def get_token(self, path_correct, tok_len=5, as_int=False):
        i = 0
        with open(path_correct,'r') as f:
            data_correct = json.load(f)

        total = min(2000,len(data_correct))

        log = os.path.join('true_tokens.csv')
        with open(log, 'w+') as f:  
            while True:      
                context = data_correct[i].get('context',data_correct[i].get('text'))
                correct_token = context[-tok_len:]
                if as_int:
                    if tok_len==5:
                        correct_token = str(token2int[correct_token])
                    else:
                        correct_token = str(gender2int[correct_token])
                f.write(correct_token+'\n')
                i += 1
                if i == total:
                    print('results saved to: ', log)
                    return

    def true_pred_int(self, pred_indices, true_tokens, dest):
        """Token as integer"""
        random_num = str(random.randint(100,900))
        c = Counter()
        with open(pred_indices, 'r') as f:
            csvread = csv.reader(f)
            pred_int = list(csvread)
    
        with open(true_tokens, 'r') as f:
            csvread = csv.reader(f)
            correct_token = list(csvread)

        log = os.path.join(dest, 'true_pred_tokens-'+random_num+'.csv')
        total = len(correct_token)
        error_cnt = 0
        with open(log, 'w+') as f:  
            for i in range(total):
                # use integer to represent token
                correct = str(token2int[correct_token[i][0]])
                pred = str(pred_int[i][0])
                if correct != pred:
                    error_cnt += 1
                f.write(correct+','+pred+'\n')
                c[correct+' | '+pred]+=1
        print('results saved to: ', log)

        counter_log = os.path.join(dest, 'counter-'+random_num+'.csv')
        with open(counter_log, 'w+') as f:
            f.write('error count: {}\n'.format(error_cnt))
            f.write('total: {}\n'.format(total))
            f.write('error%: {:.2f}%\n'.format(error_cnt/total*100))
            f.write(str(c.most_common()))
            
        print('results saved to: ', counter_log)

    def random_false(self, true_tokens, max_int=11):
        """randomly choose an incorrect token from 11 incorrect tokens"""
        with open(true_tokens, 'r') as f:
            csvread = csv.reader(f)
            correct_token = list(csvread)

        log = 'false_tokens.csv'
        with open(log, 'w+') as f:
            for row in correct_token:
                correct = str(row[0])
                # init false token to be same as correct
                false_token = str(row[0])
                while false_token==correct:
                    # change false token to be different from correct 
                    false_token = str(random.randint(0,max_int))
                f.write(false_token+'\n')

        print('results saved to: ', log)

    def random_error(self, true_tokens, false_tokens, dirname, sorted_files, stereoset=False):
        # create array to store likelihoods from each file as a row
        if stereoset:
            total = 1275
        else:
            total = 2000

        i = 0
        arr = np.empty((0,total),float)
        # int2token = {}
        for filename in sorted_files:
            print('{}: {}'.format(i, filename))
            # int2token[str(i)]=filename
            path = os.path.join(dirname,filename)
            with open(path, 'r') as f:
                # avoid newline character in likelihood list
                likelihood = f.read().splitlines()
                arr = np.vstack((arr,np.array(likelihood)))
                i+=1

        with open(true_tokens, 'r') as f:
            csvread = csv.reader(f)
            correct_token = list(csvread)

        with open(false_tokens, 'r') as f:
            csvread = csv.reader(f)
            incorrect_token = list(csvread)          

        c = Counter()
        error_cnt = 0
        log = 'true_pred.csv'
        with open(log, 'w+') as f:
            # index 2 rows from arr for correct and incorrect token          
            for i in range(total):
                correct = correct_token[i][0]
                incorrect = incorrect_token[i][0]
                correct_lld = arr[int(correct)][i]
                incorrect_lld = arr[int(incorrect)][i]
                if incorrect_lld > correct_lld:
                    error_cnt += 1
                    pred = incorrect
                else:
                    pred = correct
                f.write('{},{}\n'.format(correct, pred))
                c['{}, {}'.format(correct, pred)]+=1

        print('results saved to: ', log)

        # log results
        counter_log = 'counter.csv'
        with open(counter_log, 'w+') as f:
            f.write('error count: {}\n'.format(error_cnt))
            f.write('total: {}\n'.format(total))
            f.write('error%: {:.2f}%\n'.format(error_cnt/total*100))

            for key, value in c.items():
                f.write('{}, {}\n'.format(key, value))

        print('results saved to: ', counter_log)

if __name__ == '__main__':
    """get correct token as int"""
    # path_correct = 'my_data/genderation_data_token/convai2/dev.json'
    # path_correct = 'my_data/stereoset_token/stereoset_dev_flattened_token.json'
    # ModelError('', '', '').get_token(path_correct, tok_len=5, as_int=True)

    # path_correct = 'my_data/genderation_data/convai2/dev.json'
    # ModelError('', '', '').get_token(path_correct, tok_len=4, as_int=True)

    """get random incorrect token"""
    # true_tokens = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/stereoset_random_false/true_tokens.csv'
    # true_tokens = 'checkpoint/Reddit_90M_genderation_LRx8/target_likelihood/convai2_random_false/true_tokens.csv'
    # ModelError('', '', '').random_false(true_tokens, max_int=3)

    """
    compute model error for (random) incorrect token
    inputs: true_tokens, false_tokens, dirname of likelihood files for every token
    """
    # true_tokens = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/convai2_random_false/true_tokens.csv'
    # false_tokens = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/convai2_random_false/false_tokens.csv'
    # dirname = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/all_tokens_convai2'
    
    # true_tokens = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/stereoset_random_false/true_tokens.csv'
    # false_tokens = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/stereoset_random_false/false_tokens.csv'
    # dirname = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/all_tokens_stereoset'
    
    # true_tokens = 'checkpoint/Reddit_90M_genderation_LRx8/target_likelihood/convai2_random_false/true_tokens.csv'
    # false_tokens = 'checkpoint/Reddit_90M_genderation_LRx8/target_likelihood/convai2_random_false/false_tokens.csv'
    # dirname = 'checkpoint/Reddit_90M_genderation_LRx8/target_likelihood/all_tokens_convai2'

    # true_tokens = 'checkpoint/Reddit_90M_genderation_LRx8/target_likelihood/convai2_fixed_f0m0/true_tokens.csv'
    # false_tokens = 'checkpoint/Reddit_90M_genderation_LRx8/target_likelihood/convai2_fixed_f0m0/false_tokens.csv'
    # dirname = 'checkpoint/Reddit_90M_genderation_LRx8/target_likelihood/all_tokens_convai2'

    true_tokens = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/stereoset_fixed_f0m0u/true_tokens.csv'
    false_tokens = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/stereoset_fixed_f0m0u/false_tokens.csv'
    dirname = 'checkpoint/Reddit_90M_FT_once_genderation_token/target_posterior/all_tokens_stereoset'
    sorted_files = sorted(os.listdir(dirname))
    m = ModelError('', '', '')
    m.random_error(true_tokens, false_tokens, dirname, sorted_files, stereoset=True)