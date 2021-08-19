"""
Download Reddit pushshift data via API and preprocess data 
according to Section 6.1 of a paper by Roller et al.:
https://arxiv.org/pdf/2004.13637.pdf 
"""
import zstandard 
import json
from collections import defaultdict
import pprint
from tqdm import tqdm
from tokenizers import CharBPETokenizer
from tree_node import Node
import os
import csv
import requests
import time
import datetime

CHUNK_SIZE = 65536    

class Preprocessing:
    def __init__(self, path):
        self.path = path
        subreddit_path = './reddit/non-english-subreddit.txt'
        with open(subreddit_path) as f:
            self.subreddit_set = set(f.read().split())

        vocab = "./reddit/openai-gpt-vocab.json"
        merges = "./reddit/openai-gpt-merges.txt"
        self.tokenizer = CharBPETokenizer(vocab, merges)
        self.full_extracted = "./reddit/full_extracted"
        self.submissions = "./reddit/reddit_submissions_after_aug19_100k.json"
        self.reddit_folder = "./reddit/"
        self.reddit_raw_folder = "/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/reddit/raw_data/"
        self.reddit_preprocessed_folder = "/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/reddit/preprocessed_data/"

    def read_json(self, path):
        f = open(path)
        data = json.load(f)   
        return data['data']

    def build_forests(self, d):
        """
        Wrapper
        Build one forest per link_id dict (all comments with same link_id),
        where top level comments are tree roots.
        d is default dict with (key, value) = (link_id, [comment json_obj])
        """
        print('building forests...')
        forests = []

        for link_id, comments in tqdm(d.items()):
            # for looking up seen nodes (not necessarily tree root) for a link_id
            seen = {}
            roots = []

            # for a link_id
            for json_obj in comments:
                # build forest for comment list, each top level comment a new tree
                # order of parent before children in comment list not matter
                
                # remove 't1_' prefix in parent_id to match uid
                uid, parent_id = json_obj['id'], json_obj['parent_id'][3:]

                # check if node already added, otherwise add now
                this_node = seen.get(uid)

                if not this_node:
                    this_node = Node(uid)
                    this_node.data = json_obj
                    # this_node[data] = json_obj
                    # add to seen
                    seen[uid] = this_node
                
                else:
                    # check if added data yet
                    if not seen[uid].data:
                    # if not seen[uid]['data']:
                        # is a parent node that's added when iterating child node
                        seen[uid].data = json_obj
                        # seen[uid]['data'] = json_obj

                if json_obj['parent_id'] != link_id:
                    # not root, then should have parent
                    parent = seen.get(parent_id)
                    if not parent:
                        # create parent, if missing
                        parent = Node(parent_id)  
                        seen[parent_id] = parent
                    this_node.parent = parent
                    
            roots = [x for x in seen.values() if x.parent is None]
            print(roots)
            # print(json.dumps(roots, indent=4))

            forests.append(roots)    

        return forests

    def build_forest_from_link_id(self, link_id, verbose=False):
        """
        Build one forest per link_id (all comments with same link_id),
        where top level comments are tree roots. 
        Return list of tree roots for the link_id.
        """
        print('building forest for {}...'.format(link_id))
        link_id_path = self.reddit_raw_folder + link_id +'.json'
        comment_list = self.read_json(link_id_path)

        # for looking up seen nodes (not necessarily tree root) for a link_id
        seen = {}
        roots = []

        # build forest for comment list, each top level comment a new tree
        # order of parent before children in comment list not matter
        
        for json_obj in comment_list:
            # remove 't1_' prefix in parent_id to match uid
            uid, parent_id = json_obj['id'], json_obj['parent_id'][3:]

            # check if node already added, otherwise add now
            this_node = seen.get(uid)

            if not this_node:
                this_node = Node(uid)
                this_node.data = json_obj
                # this_node[data] = json_obj
                # add to seen
                seen[uid] = this_node
            
            else:
                # check if added data yet
                if not seen[uid].data:
                # if not seen[uid]['data']:
                    # is a parent node that's added when iterating child node
                    seen[uid].data = json_obj
                    # seen[uid]['data'] = json_obj

            if json_obj['parent_id'] != link_id:
                # not root, then should have parent
                parent = seen.get(parent_id)
                if not parent:
                    # create parent, if missing
                    parent = Node(parent_id)  
                    seen[parent_id] = parent
                this_node.parent = parent
                
        roots = [x for x in seen.values() if x.parent is None]
        if verbose:
            print('roots for link_id "{}": \n {}'.format(link_id, roots))

        return roots

    def traverse(self, forests):
        """Print comment node's id in tree traversal"""
        def printPreorder(root):      
            if root != []:
        
                # First print the data of node
                print(root['id'])
        
                # Then recur on each child
                for child in root['children']:
                    printPreorder(child)

        print('===========================Preorder traversal===================================================')
        printPreorder(forests[0][0])

    def flatten_print(self, forests):
        
        def printPreorder(root, hist):      
            if root != []:
        
                # First print the data of node
                print(root['id'])
                print('     parents: ', hist)
        
                # Then recur on each child
                for child in root['children']:
                    printPreorder(child, hist+'-->'+root['id'])

        print('===========================Preorder traversal===================================================')
        printPreorder(forests[0][0], '')

    def flatten(self, forests):
        """for each thread, filter comments and flatten into dialogue+response pairs"""
        def printPreorder(root, hist, depth):      
            if root != []:
        
                # current root node
                print(root['id'])
                
                # eliminate comments further than depth 7 in thread
                if depth > 7:
                    print('===============comment eliminated: depth > 7 ===============')
                    return

                if root.data:
                    if self.is_eliminated(root.data):
                        print(root.data)
                        print('===============comment eliminated===============')
                        return
                    print('dialogue history: ', hist)
                    print('--> label: ', root.data['body'])
                    # TODO: save dialogue hist + label to chunk
                print('==========================================================================================')

                # Then recur on each child
                for child in root['children']:
                    # print('root data: ', root.data)
                    if root.data:
                        printPreorder(child, hist+' '+root.data['body'], depth+1)
                    else:
                        printPreorder(child, hist, depth+1)

        print('===========================Preorder traversal===================================================')

        # link_id (not a comment) is depth -1, top level comment is depth 0
        printPreorder(forests[0][0], '', -1)

    def get_examples(self, forest, all_examples, all_link_ids, all_comments, verbose=False):
        """for a link_id forest, filter and flatten into dialogue-response examples"""
        def printPreorder(root, hist, depth):      
            if root != []:
                # current root node
                print(root['id'])
                
                # eliminate comments further than depth 7 in thread
                if depth > 7:
                    if verbose:
                        print('comment eliminated because depth > 7')
                    return

                if root.data:
                    if self.is_eliminated(root.data):
                        print('comment eliminated')
                        if verbose: 
                            print(root.data)
                        return
                    # if not top level comment, save dialogue hist + label
                    if hist:
                        if verbose:
                            print('dialogue history: ', hist)
                            print('--> label: ', root.data['body'])
                        all_examples.append({'text': hist, 'labels':root.data['body']})
                        all_link_ids.add(root.data['link_id'][3:])
                        all_comments.append(root.data['id'])

                # Then recur on each child
                for child in root['children']:
                    # print('root data: ', root.data)
                    if root.data:
                        printPreorder(child, hist+' '+root.data['body'], depth+1)
                    else:
                        printPreorder(child, hist, depth+1)

        # link_id (not a comment) is depth -1, top level comment is depth 0
        for root in forest:
            printPreorder(root, '', -1)

        return all_examples, all_link_ids, all_comments

    def is_bot(self, json_obj):
        """If is_bot is True, discard comment and its children"""
        author = json_obj['author']
        return author.find('bot') != -1

    def is_non_eng(self, json_obj):
        """If is_non_eng is True, discard comment and its children"""
        subreddit = json_obj['subreddit']
        return subreddit in self.subreddit_set

    def is_deleted(self, json_obj):
        """If is_deleted is True, discard comment and its children"""
        body = json_obj['body']
        return body=='[deleted]' or body=='[removed]'

    def is_gibberish(self, json_obj):
        body = json_obj['body']
        return len(body)>2048 and len(body.split())==1

    def is_long_BPE(self, json_obj):
        body = json_obj['body']
        encoded = self.tokenizer.encode(body)
        if len(encoded.tokens)>128:
            # print('Comment eliminated because is long BPE')
            return True
        else:
            return False

    def is_short(self, json_obj):
        body = json_obj['body']
        return len(body)<5

    def has_url(self,json_obj):
        body = json_obj['body']
        return body.find('http') != -1 or body.find('www') != -1     

    def is_non_ascii(self,json_obj):
        first_char = json_obj['body'][0]
        return not first_char.isascii()

    def is_eliminated(self, json_obj):
        return self.is_bot(json_obj) or self.is_non_eng(json_obj) or \
            self.is_deleted(json_obj) or self.is_gibberish(json_obj) \
            or self.is_long_BPE(json_obj) or self.is_short(json_obj) \
            or self.has_url(json_obj) or self.is_non_ascii(json_obj)

    def date_to_utc(self):
        s = '01/08/2019'
        return time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple()) # 1564614000

    def get_link_id_from_api(self):
        print('downloading link_id...')
        # after 1st August 2019
        start_date = 1564614000
        link_ids = set()
        count = 0
        # accumulate 720k comments (assume will filter out half of them with BlenderRule)
        TARGET = 720000
        # TARGET = 1000
        pbar = tqdm(total=TARGET)
        while count<TARGET:
            url = "https://api.pushshift.io/reddit/submission/search/?limit=100&after=" + str(start_date)
            r = requests.get(url)
            status = r.status_code
            if status != 200:
                print(status)
                print('skipped start date: ', start_date)
                continue
            for json_obj in r.json()['data']:
                # only keep posts with at least one comment
                num_comments = json_obj['num_comments']
                if num_comments == 0:
                    continue
                
                link_id = json_obj['id']
                link_ids.add(link_id)
                count += num_comments
                pbar.update(num_comments)

            end_date = r.json()['data'][-1]['created_utc']
            start_date = end_date + 1

        pbar.close()
        print('# link ids: ', len(link_ids))

        with open(self.reddit_folder+'link_id_list.csv', 'w+', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(link_ids)       

    def get_comments(self):
        print('downloading comments...')
        with open(self.reddit_folder+'link_id_list.csv', newline='') as f:
            reader = csv.reader(f)
            # link_ids is list of list
            link_ids = list(reader)[0]

        num_comments = []

        # start_idx = 53293 # previously: 0, 3706 (+16538), 20243, 36768, 53293
        # end_idx = len(link_ids[0]) # previously: -, -, 36768, 53293, 63647 (total)
        # print('Starting index in link_id list: ', start_idx) 
        # print('Ending index in link_id list: ', end_idx) 
        # missing = ['ckhmtg', 'ckg7a8', 'ckhrii', 'cki9hw', 'ckiayk', 'ckh4zi', 'ckhr1p']
        
        for link_id in tqdm(link_ids): # download all link_ids
        # for link_id in tqdm(link_ids[start_idx:end_idx]): # download specified range
        # for link_id in tqdm(missing): # download custom list
            url = "https://api.pushshift.io/reddit/comment/search/?limit=100&link_id=" + link_id
            r = requests.get(url)
            status = r.status_code
            if status != 200:
                print(status)
                print('sleep for 1 minute...')
                time.sleep(60)
                # try api again
                r = requests.get(url)
                status = r.status_code
                if status != 200:                
                    print('Skipped link_id: ', link_id)
                    continue
            api_data = r.json()
            count = len(api_data['data'])
            num_comments.append(count)

            with open(self.reddit_raw_folder+link_id+'.json', 'w+') as f:
                json.dump(api_data, f, indent = 4)

        print('# comments downloaded: ', sum(num_comments))

        # with open(self.reddit_folder+'num_comments_'+str(end_idx)+'.csv', 'w+', newline='') as f:
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(num_comments) 

    def get_link_id_from_file(self):
        data = self.read_json(self.submissions)
        link_ids = set()
        print('data len: ', len(data))

        for s in data:
            link_id = s['id']
            if link_id in link_ids:
                print('duplicate: ', link_id)
            link_ids.add(link_id)

        with open(self.reddit_folder+'link_id_list.csv', 'w+', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(link_ids)

        return link_ids

    def save_examples(self, all_examples):
        with open(self.reddit_preprocessed_folder+'reddit_test_2.json', 'w+') as f:
            json.dump(all_examples, f, indent = 4)
        print('# of examples: ', len(all_examples))
        print('Example data:')
        pprint.pprint(all_examples[0])

    def save_list_as_csv(self, data_list, filename):
        with open(self.reddit_preprocessed_folder+filename+'.csv', 'w+', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(data_list)         
            print('# of {}: {}'.format(filename, len(data_list)))

    def create_test_data(self):
        print('getting test data...')
        
        all_examples = []
        all_link_ids = set()
        all_comments = []

        with open(self.reddit_folder+'link_id_list.csv', newline='') as f:
            # csv file contains 1 line, comma-separated
            reader = csv.reader(f)
            # link_ids is list of list
            link_ids = list(reader)[0]       

        for link_id in tqdm(link_ids):
            print('============================================================')
            forest = self.build_forest_from_link_id(link_id)
            all_examples, all_link_ids, all_comments = self.get_examples(forest, all_examples, all_link_ids, all_comments)
        
        self.save_examples(all_examples)                
        self.save_list_as_csv(list(all_link_ids), 'filtered_link_ids')
        self.save_list_as_csv(all_comments, 'filtered_comment_ids')


if __name__ == "__main__":
    # p = Preprocessing('reddit/RC_2019-07.zst')
    # d = p.stream()

    """
    path = 'reddit/reddit_mini.json'
    p = Preprocessing(path)

    d = defaultdict(list)
    data = p.read_json(path)
    d['c7iq5c'] = data

    forests = p.build_forest(d)
    p.flatten(forests)
    """

    """
    path = 'reddit/reddit_mini.json'
    p = Preprocessing(path)
    # p.get_link_id_from_api()
    p.get_comments()
    """
    """
    path = ''
    p = Preprocessing(path)

    all_examples = []
    link_ids = ['ckhmtg', 'ckhrii']
    for link_id in link_ids:
        forest = p.build_forest_from_link_id(link_id)
        all_examples = p.get_examples(forest, all_examples)
    
    p.save_examples(all_examples)
    """

    """
    path = ''
    p = Preprocessing(path)

    all_examples = []
    all_link_ids = set()
    all_comments = []
    link_ids = ['ckhmtg', 'ckhrii']

    for link_id in link_ids:
        forest = p.build_forest_from_link_id(link_id)
        all_examples, all_link_ids, all_comments = p.get_examples(forest, all_examples, all_link_ids, all_comments) 
    
    p.save_examples(all_examples)
    p.save_list_as_csv(list(all_link_ids), 'filtered_link_ids')
    p.save_list_as_csv(all_comments, 'filtered_comment_ids')

    """
    path = ''
    p = Preprocessing(path)
    p.create_test_data()

    