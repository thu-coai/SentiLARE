# Pre-processing codes for aspect term sentiment classification
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

sentinet, gloss_embedding, gloss_embedding_norm = load_sentinet('SentiWordNet_3.0.0.txt', 'gloss_embedding.npy')


valid_split=150
polar_idx={'positive': 0, 'negative': 1, 'neutral': 2}
idx_polar={0: 'positive', 1: 'negative', 2: 'neutral'}

def parse_SemEval14(fn):
    root=ET.parse(fn).getroot()
    corpus=[]
    opin_cnt=[0]*len(polar_idx)
    term_cnt = {}
    sent_cnt = 0
    aspect_cnt = 0
    id_list, sentence_list, term_list, label_list = [], [], [], []
    for sent in root.iter("sentence"):
        sent_cnt += 1
        opins=set()
        for opin in sent.iter('aspectTerm'):
            if int(opin.attrib['from'] )!=int(opin.attrib['to'] ) and opin.attrib['term']!="NULL":
                if opin.attrib['polarity'] in polar_idx:
                    opins.add((opin.attrib['term'], int(opin.attrib['from']), int(opin.attrib['to']), opin.attrib['polarity'] ) )
        aspect_cnt += len(opins)
        for ix, opin in enumerate(opins):
            opin_cnt[polar_idx[opin[3]]]+=1
            if opin[0] not in term_cnt:
                term_cnt[opin[0]] = 1
            else:
                term_cnt[opin[0]] += 1
            cur_sent = sent.find('text').text
            id_list.append(sent.attrib['id']+"_"+str(ix))
            sentence_list.append(cur_sent)
            term_list.append(opin[0])
            label_list.append(opin[-1])

    id_list_text, text_list, text_list_split, pos_text_list, senti_text_list, label_list_text = process_text(id_list, sentence_list,
                                                                                               label_list,
                                                                                               sentinet,
                                                                                               gloss_embedding,
                                                                                               gloss_embedding_norm,
                                                                                               'discrete')

    id_list_term, term_list, term_list_split, pos_term_list, senti_term_list, label_list_term = process_text(id_list, term_list,
                                                                                               label_list,
                                                                                               sentinet,
                                                                                               gloss_embedding,
                                                                                               gloss_embedding_norm,
                                                                                               'discrete')

    assert len(id_list_text) == len(id_list_term)
    assert len(text_list) == len(term_list)
    assert len(text_list_split) == len(term_list_split)
    assert len(pos_text_list) == len(pos_term_list)
    assert len(senti_text_list) == len(senti_term_list)
    assert len(label_list_text) == len(label_list_term)

    for idx in range(len(text_list)):
        corpus.append({"id": id_list_text[idx], "sentence": text_list[idx], "term": term_list[idx], "polarity": label_list_text[idx],
                        "pos_sent": pos_text_list[idx], "senti_sent": senti_text_list[idx], "sentence_split": text_list_split[idx],
                        "pos_term": pos_term_list[idx], "senti_term": senti_term_list[idx], "term_split": term_list_split[idx]})

    print('number of sentences with each label: ', opin_cnt)
    print('number of aspect terms: ', len(term_cnt))
    print('number of ATSC samples: ', len(corpus))
    print('number of total sentences: ', sent_cnt)
    print('number of total aspects: ', aspect_cnt)
    print('\n')
    return corpus

train_corpus=parse_SemEval14('../raw_data/aspect_data/semeval14/train/Laptop_Train_v2.xml')
with open("../preprocessed_data/laptop/train_cased_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:-valid_split] }, fw, sort_keys=True, indent=4)
with open("../preprocessed_data/laptop/dev_cased_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[-valid_split:] }, fw, sort_keys=True, indent=4)
test_corpus=parse_SemEval14('../raw_data/aspect_data/semeval14/test/Laptops_Test_Gold.xml')
with open("../preprocessed_data/laptop/test_cased_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)


train_corpus=parse_SemEval14('../raw_data/aspect_data/semeval14/train/Restaurants_Train_v2.xml')
with open("../preprocessed_data/restaurant/train_cased_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:-valid_split] }, fw, sort_keys=True, indent=4)
with open("../preprocessed_data/restaurant/dev_cased_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[-valid_split:] }, fw, sort_keys=True, indent=4)

test_corpus=parse_SemEval14('../raw_data/aspect_data/semeval14/test/Restaurants_Test_Gold.xml')
with open("../preprocessed_data/restaurant/test_cased_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)
