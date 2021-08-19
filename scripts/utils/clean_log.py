"""Extract prompt and model response and clean it for RoBERTa tokenizer"""
import csv
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
    (" ... ", "... "),
]

class CleanLog:
    def __init__(self):
        pass

    def clean(self, text):
        """remove extra space around punctuation or between words"""
        for punc in PUNCTUATION_LST: 
            text = text.replace(punc[0], punc[1]) # required to isolate words
        # one space instead of 2 spaces
        text = ' '.join(text.split())
        return text

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

    def get_prompt(self, data):
        with open(data, 'r') as f:
            raw_data = json.load(f)
        suffix = " . acceptable ?"
        for ep in raw_data['data']:
            yield str(ep['tweet'])+suffix

    def process_log(self, data, path):
        """Given path to model response file, return csv file"""
        orig_data = self.get_prompt(data)
        outfile = path[:-4] + '_clean.csv'
        with open(outfile, 'w+') as f:
            # create the csv writer
            writer = csv.writer(f)
            # cleaning index
            i = 0
            for response in self.get_model_responses(path):
                if i % 2 == 0:
                    # tweet prompt
                    prompt = next(orig_data)
                else:
                    # "why" prompt
                    prompt = 'why ?'
                # clean
                prompt = self.clean(prompt)
                response = self.clean(response)
                row = prompt, response
                # print(row)
                # write a row to the csv file
                writer.writerow(row)
                i += 1

if __name__ == '__main__':
    c = CleanLog()
    data = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/my_data/ambivalent_sexism/hostile_sexism/hostile_prompt.json'
    # paths = [
    #     '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/checkpoint/Reddit_90M_genderation_LRx8/sdb/43119532.decode.hostile_prompt.log',
    #     '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/checkpoint/Reddit_90M_genderation_LRx8/sdb/43119720.decode.hostile_prompt.log',
    #     '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/checkpoint/Reddit_90M_genderation_LRx8/sdb/43119729.decode.hostile_prompt.log'
    #         ]

    # paths = [
    #     'checkpoint/Reddit_90M_genderation_LRx8/sdb/43533018.decode.hostile_prompt.log',
    #     'checkpoint/Reddit_90M_genderation_LRx8/sdb/43534773.decode.hostile_prompt_floored.log',
    #     'checkpoint/Reddit_90M_genderation_LRx8/sdb/43535124.decode.hostile_prompt_floored.log',
    #     'checkpoint/Reddit_90M_genderation_LRx8/sdb/43535137.decode.hostile_prompt_floored.log',
    # ]

    # paths = [
    #     'checkpoint/Blender_90M_orig_zoo/sdb/43596186.decode.hostile_prompt.log',
    #     'checkpoint/Blender_90M_orig_zoo/sdb/43607095.decode.hostile_prompt.log',
    #     'checkpoint/Blender_90M_orig_zoo/sdb/43609533.decode.hostile_prompt.log',
    #     'checkpoint/Blender_90M_orig_zoo/sdb/43609545.decode.hostile_prompt.log'
    # ]

    paths = [
        'checkpoint/Reddit_90M_FT_once_genderation_token/sdb/43713539.decode.hostile_prompt.log',
        'checkpoint/Reddit_90M_FT_once_genderation_token/sdb/43714083.decode.hostile_prompt.log',
        'checkpoint/Reddit_90M_FT_once_genderation_token/sdb/43714120.decode.hostile_prompt.log',
        'checkpoint/Reddit_90M_FT_once_genderation_token/sdb/43714129.decode.hostile_prompt.log',
    ]
    for path in paths:
        c.process_log(data, path)