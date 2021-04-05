# Pre-processing codes for aspect category detection
# This code is modified based on
# https://github.com/howardhsu/BERT-for-RRC-ABSA/blob/master/pytorch-pretrained-bert/preprocessing/prep_asc.py

import nltk
import numpy as np
import json
from collections import defaultdict
import xml.etree.ElementTree as ET
import random
random.seed(1337)
np.random.seed(1337)
from nltk.tokenize import word_tokenize
from aspect_utils import process_text, load_sentinet

valid_split=150
sem14_category = {'food':0, 'service':1, 'price':2, 'ambience':3, 'anecdotes/miscellaneous':4}
sem16_category = {'RESTAURANT#GENERAL':0, 'RESTAURANT#PRICES':1, 'RESTAURANT#MISCELLANEOUS':2,
                  'FOOD#PRICES':3, "FOOD#QUALITY":4, "FOOD#STYLE_OPTIONS":5, "DRINKS#PRICES": 6,
                  "DRINKS#QUALITY":7, "DRINKS#STYLE_OPTIONS":8, "AMBIENCE#GENERAL":9, "SERVICE#GENERAL": 10,
                  "LOCATION#GENERAL": 11}

sentinet, gloss_embedding, gloss_embedding_norm = load_sentinet('SentiWordNet_3.0.0.txt', 'gloss_embedding.npy')

def parse_SemEval14(fn):
    root=ET.parse(fn).getroot()
    corpus=[]
    term_cnt = {}
    sent_cnt = 0
    aspect_cnt = 0
    id_list, sentence_list, term_list = [], [], []
    for sent in root.iter("sentence"):
        sent_cnt += 1
        opins=set()
        for opin in sent.iter('aspectCategory'):
            if opin.attrib['category'] != "NULL":
                opins.add(opin.attrib['category'])
        aspect_cnt += len(opins)
        category_list = []
        for ix, opin in enumerate(opins):
            category_list.append(opin)
            if opin not in term_cnt:
                term_cnt[opin] = 1
            else:
                term_cnt[opin] += 1

        cur_sent = sent.find('text').text
        id_list.append(sent.attrib['id'])
        sentence_list.append(cur_sent)
        term_list.append(category_list)

    clean_id_list, text_list, text_list_split, pos_list, senti_list, label_list = process_text(id_list, sentence_list, term_list,
                                                                                sentinet, gloss_embedding, gloss_embedding_norm, 'discrete')

    for idx in range(len(text_list)):
        corpus.append({"id": clean_id_list[idx], "sentence": text_list[idx], "term": label_list[idx],
                        "pos_sent": pos_list[idx], "senti_sent": senti_list[idx], "sentence_split": text_list_split[idx]})

    print('number of sentences with each label: ', term_cnt)
    print('number of training samples: ', len(corpus))
    print('number of total sentences: ', sent_cnt)
    print('number of total aspects: ', aspect_cnt)
    return corpus


def parse_SemEval16(fn):
    root=ET.parse(fn).getroot()
    corpus=[]
    term_cnt = {}
    aspect_cnt = 0
    sent_cnt = 0
    id_list, sentence_list, term_list = [], [], []
    for review in root.iter("Review"):
        for sents in review.iter("sentences"):
            for sent in sents.iter("sentence"):
                sent_cnt += 1
                opins=set()
                for opin_s in sent.iter('Opinions'):
                    for opin in opin_s.iter('Opinion'):
                        if opin.attrib['category'] != "NULL":
                            opins.add(opin.attrib['category'])
                aspect_cnt += len(opins)
                category_list = []
                for ix, opin in enumerate(opins):
                    category_list.append(opin)
                    if opin not in term_cnt:
                        term_cnt[opin] = 1
                    else:
                        term_cnt[opin] += 1

                cur_sent = sent.find('text').text
                id_list.append(sent.attrib['id'])
                sentence_list.append(cur_sent)
                term_list.append(category_list)

    clean_id_list, text_list, text_list_split, pos_list, senti_list, label_list = process_text(id_list, sentence_list,
                                                                                               term_list,
                                                                                               sentinet,
                                                                                               gloss_embedding,
                                                                                               gloss_embedding_norm,
                                                                                               'discrete')

    for idx in range(len(text_list)):
        corpus.append({"id": clean_id_list[idx], "sentence": text_list[idx], "term": label_list[idx],
                       "pos_sent": pos_list[idx], "senti_sent": senti_list[idx], "sentence_split": text_list_split[idx]})

    print('number of sentences with each label: ', term_cnt)
    print('number of training samples: ', len(corpus))
    print('number of total sentences: ', sent_cnt)
    print('number of total aspects: ', aspect_cnt)

    return corpus


train_corpus=parse_SemEval14('../raw_data/aspect_data/semeval14/train/Restaurants_Train_v2.xml')
with open("../preprocessed_data_acd/res14/train_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:-valid_split] }, fw, sort_keys=True, indent=4)
with open("../preprocessed_data_acd/res14/dev_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[-valid_split:] }, fw, sort_keys=True, indent=4)
test_corpus=parse_SemEval14('../raw_data/aspect_data/semeval14/test/Restaurants_Test_Gold.xml')
with open("../preprocessed_data_acd/res14/test_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)

train_corpus=parse_SemEval16('../raw_data/aspect_data/semeval16/sentence/train/ABSA16_Restaurants_Train_SB1_v2.xml')
with open("../preprocessed_data_acd/res16/train_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:-valid_split] }, fw, sort_keys=True, indent=4)
with open("../preprocessed_data_acd/res16/dev_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[-valid_split:] }, fw, sort_keys=True, indent=4)
test_corpus=parse_SemEval16('../raw_data/aspect_data/semeval16/sentence/test/EN_REST_SB1_TEST.xml.gold')
with open("../preprocessed_data_acd/res16/test_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)
