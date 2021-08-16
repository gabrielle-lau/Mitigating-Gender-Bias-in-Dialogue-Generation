"""
Download and clean hostile sexism Twitter dataset
Reference: https://stackoverflow.com/questions/28384588/twitter-api-get-tweets-with-specific-id 

Sample usage
```
python preprocess_tweets.py hostile_sexist.tsv \
    my_data/ambivalent_sexism/hostile_data.csv &>> tweet_scraping.log
```

Retrieve tweet from id using curl
```
curl "https://api.twitter.com/2/tweets/[tweet id]" -H "Authorization: Bearer [bearer token]"
```

Tweet URL on Twitter
https://twitter.com/i/web/status/[tweet id]
"""
from __future__ import print_function
import getopt
import os
import sys
import tweepy
import csv  
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import string
import re
import ast
import emoji
# demoji.download_codes() # Run this once 
import demoji
import json

DATA_FOLDER = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/ambivalent_sexism/'
RESULT_FILE = None
PUNC = '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' # modified string.punctuation

# Get tokens by clicking the key icon at https://apps.twitter.com/app
CONSUMER_KEY = '[Api key (username)]'
CONSUMER_SECRET = '[Secret key (password)]'
OAUTH_TOKEN = '[Access token]'
OAUTH_TOKEN_SECRET = '[Access token secret]'

# batch size depends on Twitter limit, 100 at this time
batch_size=100

def get_tweet_id(line):
    '''
    Extracts and returns tweet ID from a line in the input.
    '''
    # (tagid,_timestamp,_sandyflag) = line.split('\t')
    # (_tag, _search, tweet_id) = tagid.split(':')
    tweet_id = line[0] # ['839880162586071040']
    return tweet_id

def get_tweet_list(twapi, idlist):
    '''
    CHANGE: Clean before writing to csv, don't encode utf-8
    Invokes bulk lookup method.
    Raises an exception if rate limit is exceeded.
    
    '''
    # fetch as little metadata as possible
    # tweets = twapi.statuses_lookup(id_=idlist, include_entities=False, trim_user=True, tweet_mode="extended")
    tweets = twapi.statuses_lookup(id_=idlist, tweet_mode="extended")
    if len(idlist) != len(tweets):
        print('get_tweet_list: unexpected response size %d, expected %d', len(tweets), len(idlist))

    # Open/create a file to append data to
    csvFile = open(RESULT_FILE, 'a')

    #Use csv writer
    csvWriter = csv.writer(csvFile)

    for tweet in tweets:
        # tweet_text = tweet.text.encode('UTF-8') # problem: truncated to 140 char, switch to extended mode and use "full_text"
        tweet_text = tweet.full_text
        try:
            # is a truncated retweet, override by original tweet's full text
            # print('retweeted full_text: ', tweet.retweeted_status.full_text)
            print('Retweet:')
            tweet_text = tweet.retweeted_status.full_text
        except:
            # not a retweet
            pass
        print('%s,%s' % (tweet.id, tweet_text))
        tweet_text = clean_ascii(tweet_text)
        csvWriter.writerow([tweet.id, tweet_text])
        print('%s,%s' % (tweet.id, tweet_text))

def get_tweets_bulk(twapi, idfilepath):
    '''
    Fetches content for tweet IDs in a file using bulk request method,
    which vastly reduces number of HTTPS requests compared to above;
    however, it does not warn about IDs that yield no tweet.

    `twapi`: Initialized, authorized API object from Tweepy
    `idfilepath`: Path to file containing IDs
    '''    
    # process IDs from the file
    tweet_ids = list()

    
    with open(idfilepath, 'r') as f:
        idfile = csv.reader(f, delimiter="\t", quotechar='"')
        pbar = tqdm(total=len(list(idfile)))

    # with open(idfilepath, 'rb') as idfile:
    with open(idfilepath, 'r') as f:
        idfile = csv.reader(f, delimiter="\t", quotechar='"')
        
        for line in idfile:
            tweet_id = get_tweet_id(line)
            tweet_ids.append(tweet_id)
            # API limits batch size
            if len(tweet_ids) == batch_size:
                get_tweet_list(twapi, tweet_ids)
                tweet_ids = list()
                pbar.update(batch_size)
    pbar.close()
    # process remainder
    if len(tweet_ids) > 0:
        get_tweet_list(twapi, tweet_ids)

def remove_duplicate(df):
    df = df.drop_duplicates(subset='tweet')
    return df

def clean_ascii(tweet):
    """clean ascii, unencoded tweet"""
    if tweet[0:2] == 'RT':
        tweet = tweet[2:]
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @+username tags
    tweet = re.sub("#[A-Za-z0-9]+","",tweet) #Remove hashtags
    tweet = re.sub("&amp","and",tweet) #Replace "&amp" with "and"
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())  # remove extra space to prepare for punctuation removal
    tweet = ' '.join(word.strip(PUNC).lower() for word in tweet.split()) # remove punctuation but not aprostrophe/hashtag, lower case
    tweet = demoji.replace(tweet, "") #Remove Emojis   
    return tweet

def preprocess_duplicate(path, output_name):
    df = pd.read_csv(path, names=['id','tweet'])
    df = remove_duplicate(df)
    df = df.reset_index() # count index ignoring the removed rows
    result = df.to_json(orient="table")
    parsed = json.loads(result)
    with open(DATA_FOLDER+output_name+'.json', 'w+') as f:
        json.dump(parsed, f, indent=4)

# def main(args):
def main(id_file, output_file):
    print('-----------------------------------')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print('id_file: ', id_file)
    print('output_file: ', output_file)

    bulk = True
    idfile = id_file 
    if not os.path.isfile(idfile):
        print('Not found or not a file: %s' % idfile, file=sys.stderr)
        usage()

    # connect to twitter
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)

    get_tweets_bulk(api, idfile)

if __name__ == '__main__':
    # download and clean tweets simultaneously
    id_file = sys.argv[1]
    RESULT_FILE = sys.argv[2]
    main(id_file, RESULT_FILE)
    
    # clean downloaded tweets
    path = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/ambivalent_sexism/hostile_data.csv'
    # just remove duplicates
    preprocess_duplicate(path, 'hostile_data_cleaned')

    # manually added space around ellipsis
    # manually removed usernames with underscore