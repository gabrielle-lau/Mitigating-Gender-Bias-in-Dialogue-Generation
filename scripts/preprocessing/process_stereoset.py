"""
Prepare StereoSet
"""

import csv
import json
from sklearn.model_selection import train_test_split
import pprint
import copy
from tqdm import tqdm

SEED = 8

class ProcessStereoset:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.dest = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/stereoset/'

    def load_data(self, path):
        """"load stereoset intersentence data from path & return list"""
        with open(path,'r') as f:
            all_data=json.load(f)
        # intersentence
        return all_data['data']['intersentence'] # list of 2123 examples

    def split_index(self, num_data=2123):
        """split by ratio 80:20 (# 1698:425) and save indices to csv"""
        X_train, X_test = train_test_split(list(range(num_data)), test_size=0.2, random_state=SEED)
        with open(self.dest+'train_indices.csv', 'w', newline='') as f:
            # 1 item per line
            wr = csv.writer(f, quoting=csv.QUOTE_NONE, delimiter='\n')
            wr.writerow(X_train)
        with open(self.dest+'test_indices.csv', 'w', newline='') as f:
            # 1 item per line
            wr = csv.writer(f, quoting=csv.QUOTE_NONE, delimiter='\n')
            wr.writerow(X_test)

    def split_json(self):
        """split dev.json into train.json and test.json"""
        inter_data = self.load_data(self.path)
        X_train, X_test = train_test_split(inter_data, test_size=0.2, random_state=SEED)
        train = {"version": "2.0-train", "data": {"intersentence": X_train}}
        test = {"version": "2.0-test", "data": {"intersentence": X_test}}
        self.save_json(train, name="stereoset_train")
        self.save_json(test, name="stereoset_test")
    
    def save_json(self, data_dict, name="my_data"):
        """save Python dict or list of dicts as json file"""
        with open(self.dest+name+'.json', 'w+') as f:
            json.dump(data_dict, f, indent = 4)
        print('Successfully saved: ', self.dest+name+'.json')
        # print('# of examples: ', len(data_dict['data']['intersentence']))
        # print('Example data:')
        # pprint.pprint(data_dict['data']['intersentence'][0])
        print('# of examples: ', len(data_dict))
    
    def create_train_test(self):
        """split orig stereoset into train.json and test.json of same format"""
        self.split_index(num_data=2123)
        self.split_json()        

    def read_csv(self):
        """read csv file where each line is an element into a list"""
        target_list = []
        with open(self.csv) as f:
            for row in csv.reader(f):
                # row[0] takes column 0
                target_list.append(row[0])
        print('len of target_list: ', len(target_list))
        return target_list

    def flatten_ordered(self, path, name='stereoset_train_flattened'):
        """order by anti-stereotype, stereotype, unrelated for every three examples with identical contexts """
        raw_data = self.load_data(path)
        results = []
        for context_dict in tqdm(raw_data):
            # context_copy = copy.deepcopy(context_dict) # prevent modify in place
            # context_copy = copy.copy(context_dict) # prevent modify in place
            # context_copy.pop('sentences', None)
            order = [None, None, None]
            for sentence_dict in context_dict['sentences']:
                # re-initialise dict, otherwise overwrite those already in results
                example_dict = {
                'context': context_dict['context'],
                'context_id': context_dict['id'],
                'target': context_dict['target'],
                'bias_type': context_dict['bias_type']
                }
                example_dict['sentence'] = sentence_dict['sentence']
                example_dict['gold_label'] = sentence_dict['gold_label']
                example_dict['sentence_id'] = sentence_dict['id']
                # pprint.pprint(example_dict)

                if sentence_dict['gold_label'] == 'anti-stereotype':
                    index = 0
                elif sentence_dict['gold_label'] == 'stereotype':
                    index = 1
                else:
                    index = 2
                order[index] = example_dict
            results += order
            
        self.save_json(results, name=name)        

    def unique_data(self, path, name):
        """unique prompt, not explode"""
        raw_data = self.load_data(path)
        results = []
        for context_dict in tqdm(raw_data):
            example_dict = {
                'context': context_dict['context'],
                'context_id': context_dict['id'],
                'target': context_dict['target'],
                'bias_type': context_dict['bias_type'],
                'sentence': ['', '', ''],
                'sentence_id': ['', '', '']
                }
            for sentence_dict in context_dict['sentences']:
                # re-initialise dict, otherwise overwrite those already in results
                if sentence_dict['gold_label'] == 'anti-stereotype':
                    index = 0
                elif sentence_dict['gold_label'] == 'stereotype':
                    index = 1
                else:
                    index = 2
                example_dict['sentence'][index] = sentence_dict['sentence']
                example_dict['sentence_id'][index] = sentence_dict['id'],
            results.append(example_dict)
            # pprint.pprint(example_dict)
        self.save_json(results, name=name)

if __name__ == '__main__':
    # original stereoset data
    path = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/stereoset/data/dev.json'
    
    p = ProcessStereoset(path)
    ## split train test
    # p.create_train_test()

    ## flatten hierarchical structure to example pairs ordered by a,s,u
    train_path = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/stereoset/orig(non-task)/stereoset_train.json'
    test_path = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/stereoset/orig(non-task)/stereoset_test.json'
    p.flatten_ordered(train_path)
    p.flatten_ordered(test_path, name='stereoset_test_flattened')