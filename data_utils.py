import torch
import pandas as pd
import numpy as np
import csv
import spacy
import os
import re
from torchtext import data, datasets
import argparse

nlp = spacy.load('en_core_web_sm')

def read_semeval_data(path,name):
    filename = os.path.join(path,name)
    data = {'relation':[], 'sentence':[], 'pos_embed':[]}
    etags = ['<e1>','</e1>','<e2>','</e2>']
    with open(filename, 'r') as rf:
        for line in rf:
            try:
                _, sent = line.split('\t')
                sent = sent.strip().lower()[1:-1]

                rel = next(rf).strip().upper()
                next(rf) # comment
                next(rf) # blankline
                e1 = sent[sent.index('<e1>')+4:sent.index('</e1>')]
                if ' ' in e1 or '-' in e1:
                    e1 = re.split('[ -]',e1)[0]

                e2 = sent[sent.index('<e2>')+4:sent.index('</e2>')]
                if ' ' in e2 or '-' in e2:
                    e2 = re.split('[ -]',e2)[0]

                sent = re.sub('<(/)*e[1-2]>',' ',sent)
                sent = re.sub('\s+',' ',sent)
                words = [tok.text for tok in nlp.tokenizer(sent)]

                index1 = words.index(e1)
                index2 = words.index(e2)
                pos_embed = [[i-index1, i-index2] for i in range(len(words))]

                data['relation'].append(rel)
                data['sentence'].append(sent)
                data['pos_embed'].append(pos_embed)
            except Exception as e:
                print(e)
                print(sent)
                print(e1)
                print(e2)
                print(words)
    df = pd.DataFrame.from_dict(data)
    return df

indexs = np.random.permutation(8000)

path = '../data/SemEval2010_task8_all_data/SemEval2010_task8_training'
df = read_semeval_data(path,'TRAIN_FILE.TXT')
newName = 'TRAIN_FILE_SUB.CSV'
newFile = os.path.join(path,newName)
df1 = df.iloc[indexs[:7000],:]
df1.to_csv(newFile,index=None,sep='\t')

newName = 'VALID_FILE.CSV'
newFile = os.path.join(path,newName)
df2 = df.iloc[indexs[7000:],:]
df2.to_csv(newFile,index=None,sep='\t')

path = '../data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys'
df = read_semeval_data(path,'TEST_FILE_FULL.TXT')
newName = 'TEST_FILE_FULL.CSV'
newFile = os.path.join(path,newName)
df.to_csv(newFile,index=None,sep='\t')