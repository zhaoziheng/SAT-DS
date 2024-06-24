import pandas as pd
import json
from tqdm import tqdm
import copy
from pathlib import Path
import os
import random
import numpy as np

TRAIN_RATIO = 0.8

def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False
       
def predefined_train_test_split(jsonl2load, train_jsonl, test_jsonl, split_json):
    """
    split by json
    """
    with open(split_json, 'r') as f:
        split = json.load(f)   # should be a dict with {'train':[xxx, xxx, ...], 'val':[...], 'test':[...]}
    train_ids = split['train']
    test_ids = split['test']
    
    with open(jsonl2load, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    
    train_samples = []
    test_samples = []
    for datum in lines:
        if datum['patient_id'] in train_ids:
            train_samples.append(datum)
        elif datum['patient_id'] in test_ids:
            test_samples.append(datum)
        else:
            print(f'unclassified samples : {datum["image"]}')
    
    with open(train_jsonl, 'w') as f:
        for datum in train_samples:
            f.write(json.dumps(datum)+'\n')
            
    with open(test_jsonl, 'w') as f:
        for datum in test_samples:
            f.write(json.dumps(datum)+'\n')   
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl2split')
    parser.add_argument('--train_jsonl')
    parser.add_argument('--test_jsonl')
    parser.add_argument('--split_json')
    config = parser.parse_args()

    predefined_train_test_split(config.jsonl2split, config.train_jsonl, config.test_jsonl, config.split_json)