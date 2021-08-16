"""
Input: file containing likelihoods in order of a,s,u
Return log-likliehood ratio bias score

Sample usage:
```
python bias_score.py likelihood.log outfile
```
"""
import os
from datetime import datetime
import math
import csv
import json
import sys
import pickle

GENDER_INDICES = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/stereoset/gender_indices.json'

class BiasScore:
    def __init__(self, likelihood, dest):
        super().__init__()
        # init path to likelihood file and output destination folder
        self.likelihood = likelihood
        self.dest = dest

    def log_ratio(self):
        """Return bias score"""
        with open(self.likelihood, 'r') as f:
            likelihood = f.readlines() 

        total = len(likelihood)
        print('total lines:', total)
        cnt = 0
        results = []
        for i in range(0, total, 3):
            # log ratio of log likelihoods
            # print(i, likelihood[i])
            if float(likelihood[i])==0 or float(likelihood[i+1])==0:
                print('Error: likelihood=0, skip pair # ', i)
                continue
            a_log_likelihood = math.log(float(likelihood[i]))
            s_log_likelihood = math.log(float(likelihood[i+1]))
            bias = math.log(s_log_likelihood/a_log_likelihood)
            results.append(bias)
            cnt+=1
        
        timestamp = datetime.now().strftime("%m-%d-%H")

        # bias_score = os.path.join(self.dest, 'log_ratio_bias_'+timestamp+'.csv')
        bias_score = os.path.join(self.dest + 'scores.csv')
        with open(bias_score, 'w', newline='') as f:
            # 1 item per line
            wr = csv.writer(f, quoting=csv.QUOTE_NONE, delimiter='\n')
            wr.writerow(results)

        # log = os.path.join(self.dest, 'log_ratio_bias_'+timestamp)
        log = os.path.join(self.dest + 'avg.log')
        with open(log, 'w+') as f:
            f.write(('Target likelihood: {} \nAverage bias score: {} \n'
                    '# pairs: {} \n').format(self.likelihood, sum(results)/cnt, cnt))
        
        print(('Target likelihood: {} \nAverage bias score: {} \n'
                    '# pairs: {} \n').format(self.likelihood, sum(results)/cnt, cnt))

        print('Full results saved to: ', bias_score)
        print('Summary log saved to:', log)
        print('-------------------------------------------------')

    def gender_index(self, path):
        """Get list of indices of gender bias exampels in dataset from path"""
        with open(path,'r') as f:
            all_data=json.load(f)

        indices=[]
        for i in range(0,len(all_data),3):
            if all_data[i]['bias_type']=='gender':
                indices.append(i)
        return indices
    
    def save_list(self, path, mylist):
        with open(path, "w") as f:   
            json.dump(mylist, f, indent=2)

    def load_list(self, path):
        with open(path, "r") as f:   
            mylist = json.load(f)       
        return mylist
        
    def log_ratio_gender(self):
        """Return gender specific bias score"""
        with open(self.likelihood, 'r') as f:
            likelihood = f.readlines() 

        indices=self.load_list(GENDER_INDICES)
        print('# gender bias examples: ', len(indices))
        total = len(likelihood)
        print('# total examples:', total)
        cnt = 0
        results = []
        for i in indices:
            # log ratio of log likelihoods
            # print(i, likelihood[i])
            if float(likelihood[i])==0 or float(likelihood[i+1])==0:
                print('Error: likelihood=0, skip pair # ', i)
                continue
            a_log_likelihood = math.log(float(likelihood[i]))
            s_log_likelihood = math.log(float(likelihood[i+1]))
            bias = math.log(s_log_likelihood/a_log_likelihood)
            results.append(bias)
            cnt+=1
        
        timestamp = datetime.now().strftime("%m-%d-%H")

        # bias_score = os.path.join(self.dest, 'log_ratio_bias_gender_'+timestamp+'.csv')
        bias_score = os.path.join(self.dest + 'gender_scores.csv')
        with open(bias_score, 'w', newline='') as f:
            # 1 item per line
            wr = csv.writer(f, quoting=csv.QUOTE_NONE, delimiter='\n')
            wr.writerow(results)

        # log = os.path.join(self.dest, 'log_ratio_bias_gender_'+timestamp)
        log = os.path.join(self.dest + 'gender_avg.log')
        with open(log, 'w+') as f:
            f.write(('Target likelihood: {} \nAverage bias score (gender only): {} \n'
                    '# pairs: {} \n').format(self.likelihood, sum(results)/cnt, cnt))
        
        print(('Target likelihood: {} \nAverage bias score (gender only): {} \n'
                    '# pairs: {} \n').format(self.likelihood, sum(results)/cnt, cnt))

        print('Full results saved to: ', bias_score)
        print('Summary log saved to:', log)
        print('-------------------------------------------------')

if __name__ == '__main__':
    likelihood = sys.argv[1]
    dest = sys.argv[2]
    b = BiasScore(likelihood, dest)
    b.log_ratio()
    b.log_ratio_gender()   