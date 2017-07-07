import re
import os
import sys
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import pandas as pd

def load_data():
    with open('SQuAD/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    with open('SQuAD/data.msgpack', 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train_orig = pd.read_csv('SQuAD/train.csv')
    dev_orig = pd.read_csv('SQuAD/dev.csv')
    train = list(zip(
        data['trn_context_ids'],
        data['trn_question_ids'],
        data['trn_context_tokens'],
        data['trn_question_tokens'],
        data['trn_context_text'],
        data['trn_question_text']
    ))
    dev = list(zip(
        data['dev_context_ids'],
        data['dev_question_ids'],
        data['dev_context_tokens'],
        data['dev_question_tokens'],
        data['dev_context_text'],
        data['dev_question_text']
    ))
    dev_y = dev_orig['answers'].tolist()[:len(dev)]
    dev_y = [eval(y) for y in dev_y]
    return train, dev, dev_y

train, dev, dev_y = load_data()

def write_list(filename, data):
    with open(filename, "w") as f:
        for item in data:
            if isinstance(item, list):
                item = " ".join([str(x) for x in item])
            print(item, file=f)

def write_set(which, data):
    context_ids, question_ids, context_tokens, question_tokens, context_text, question_text = zip(*data)
    write_list(which + "_context.ids", context_ids)
    write_list(which + "_question.ids", question_ids)
    write_list(which + "_context.tok", context_tokens)
    write_list(which + "_question.tok", question_tokens)
    write_list(which + "_context.txt", context_text)
    write_list(which + "_question.txt", question_text)

write_set("train", train)
write_set("dev", dev)
