"""Compare RoBERTa classification with human-annotated gold labels"""
import csv 
import pandas as pd
from collections import Counter

class CompareHuman:
    def __init__(self,path):
        # path to file with RoBERTa and human predictions
        self.path = path

    def csv2df(self, path):
        df = pd.read_csv(path)
        return df

    def compare(self, rows=200):
        df = self.csv2df(self.path)
        # only rows specified 
        human_pred = df['human_pred_int'].tolist()[:rows]
        roberta_pred = df['prediction_int'].tolist()[:rows]
        # compare
        truth = [int(i)==int(j) for i, j in zip(human_pred, roberta_pred)]
        num_true = sum(truth)
        # print results
        print('human pred counts:\n', Counter(human_pred))
        print('roberta pred counts:\n', Counter(roberta_pred))
        print('#correct: {}\n#rows checked: {}\n%correct: {:.2f}'.format(
            num_true,
            rows,
            100*num_true/rows
        ))
        # move roberta pred to second last column
        prediction_int = df['prediction_int']
        df.drop(labels=['prediction_int'], axis=1,inplace = True)
        df['prediction_int'] = prediction_int
        # add compare column to last column
        filler = ''
        # df['correct_pred'] = truth
        df.loc[:, 'correct_pred'] = truth + [filler]*(len(df.index) - len(truth))
        # export csv
        outfile = self.path[:-4] + '_compare.csv'
        df.to_csv(outfile, index=False)
        print('saved to: ', outfile)
        return None
    
if __name__ == "__main__":
    # path = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/checkpoint/Reddit_90M_genderation_LRx8/sdb/human_pred_mc/43118567.decode.hostile_prompt_robertamc_humanpred.csv'
    # path = '/rds/project/rds-xyBFuSj0hm0/myl40/mphil_project/checkpoint/Reddit_90M_genderation_LRx8/sdb/human_pred_classifier/43118567.decode.hostile_prompt_robertaclassifier_humanpred.csv'
    # path = 'checkpoint/Blender_90M_orig_zoo/sdb/human_pred/43596186.decode.hostile_prompt_clean_pred.csv'
    # path = 'checkpoint/Blender_90M_orig_zoo/sdb/human_pred/43596186.decode.hostile_prompt_clean_pred2.csv'
    # path = 'checkpoint/Blender_90M_orig_zoo/sdb/human_pred/43596186.decode.hostile_prompt_clean_humanpred_strict_idk.csv'
    # path = 'checkpoint/Reddit_90M_genderation_LRx8/sdb/human_pred_mc/agree_mc/43118567.decode.hostile_prompt_clean_humanpred.csv'
    # path = 'checkpoint/Blender_90M_orig_zoo/sdb/human_pred/acceptable_mc/43596186.decode.hostile_prompt_clean_pred_idk_strict.csv'
    # path = 'checkpoint/Blender_90M_orig_zoo/sdb_eval/human_pred/none.43693393_clean_pred.csv'
    path = 'checkpoint/Reddit_90M_FT_once_genderation_token/sdb/human_pred/43713539.decode.hostile_prompt_clean_pred.csv'
    c = CompareHuman(path)
    c.compare()

