import torch
import pandas as pd
import numpy as np
import csv
import spacy
import os
import re
from torchtext import data, datasets
import argparse
import train as trains
import model
import datetime

print('parse arguments.')
parser = argparse.ArgumentParser(description='CRCNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.025, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=300, help='number of epochs for train [default: 16]')
parser.add_argument('-batch-size', type=int, default=100, help='batch size for training [default: 256]')
parser.add_argument('-log-interval',  type=int, default=100,   help='how many steps to wait before logging training status [default: 500]')
parser.add_argument('-dev-interval', type=int, default=300, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=2000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.75, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=500, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=2, help='device to use for iterate data, -1 mean cpu [default: -1]')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()): 
    print("\t{}={}".format(attr.upper(),value))

args.sent_len = 90
args.class_num = 19
args.pos_dim = 90
args.mPos = 2.5
args.mNeg = 0.5
args.gamma = 0.05
# args.device = torch.device(args.device)
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

nlp = spacy.load('en_core_web_sm')

def tokenizer(text): # create a tokenizer function
    # 返回 a list of <class 'spacy.tokens.token.Token'>
    return [tok.text for tok in nlp.tokenizer(text)]

def emb_tokenizer(l):
    r = [y for x in eval(l) for y in x]
    return r

TEXT = data.Field(sequential=True, tokenize=tokenizer,fix_length=args.sent_len)
LABEL = data.Field(sequential=False, unk_token='OTHER')
POS_EMB = data.Field(sequential=True,unk_token=0,tokenize=emb_tokenizer,use_vocab=False,pad_token=0,fix_length=2*args.sent_len)

print('loading data...')
train,valid,test = data.TabularDataset.splits(path='../data/SemEval2010_task8_all_data',
                                              train='SemEval2010_task8_training/TRAIN_FILE_SUB.CSV',
                                              validation='SemEval2010_task8_training/VALID_FILE.CSV',
                                              test='SemEval2010_task8_testing_keys/TEST_FILE_FULL.CSV',
                                              format='csv',
                                              skip_header=True,csv_reader_params={'delimiter':'\t'},
                                              fields=[('relation',LABEL),('sentence',TEXT),('pos_embed',POS_EMB)])
TEXT.build_vocab(train,vectors='glove.6B.300d')
LABEL.build_vocab(train)

args.vocab = TEXT.vocab
args.cuda = torch.cuda.is_available()
# args.cuda = False
args.save_dir = os.path.join(args.save_dir,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

train_iter, val_iter, test_iter = data.Iterator.splits((train,valid,test),
                                                             batch_sizes=(args.batch_size,len(valid),len(test)),
                                                             device=args.device,
                                                             sort_key=lambda x: len(x.sentence),
#                                                              sort_within_batch=False,
                                                             repeat=False)

print('build model...')
cnn = model.CRCNN(args)
if args.snapshot is not None:
    print('\nLoding model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

if args.test:
    try:
        trains.eval(test_iter,cnn,args)
    except Exception as e:
        print("\n test wrong.")
else:
    trains.train(train_iter,val_iter,cnn,args)