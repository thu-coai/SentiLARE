# Pre-processing codes for aspect term extraction
# This code is modified based on
# https://github.com/howardhsu/BERT-for-RRC-ABSA/blob/master/pytorch-pretrained-bert/preprocessing/prep_ae.py

import nltk
import numpy as np
import json
from collections import defaultdict
import xml.etree.ElementTree as ET
import random
random.seed(1337)
np.random.seed(1337)
from aspect_utils import process_text, load_sentinet

valid_split=150
polar_idx={'positive': 0, 'negative': 1, 'neutral': 2, 'conflict': 3}
idx_polar={0: 'positive', 1: 'negative', 2: 'neutral'}

sentinet, gloss_embedding, gloss_embedding_norm = load_sentinet('SentiWordNet_3.0.0.txt', 'gloss_embedding.npy')


def parse_SemEval14(fn):
    root=ET.parse(fn).getroot()
    corpus = []
    opin_cnt=[0]*len(polar_idx)
    sent_cnt = 0
    aspect_cnt = 0
    id_list, sentence_list, term_list = [], [], []
    for sent in root.iter("sentence"):
        text=[]
        opins=set()
        sent_cnt += 1
        for opin in sent.iter('aspectTerm'):
            if int(opin.attrib['from'] )!=int(opin.attrib['to'] ) and opin.attrib['term']!="NULL":
                opins.add((opin.attrib['term'], int(opin.attrib['from']), int(opin.attrib['to']), polar_idx[opin.attrib['polarity'] ] ) )

        aspect_cnt += len(opins)
        for ix, c in enumerate(sent.find('text').text ):
            for opin in opins:
                if (c=='/' or c=='*' or c=='-' or c=='=') and len(text)>0 and text[-1]!=' ':
                    text.append(' ')
                if ix==int(opin[1] ) and len(text)>0 and text[-1]!=' ':
                    text.append(' ')
                elif ix==int(opin[2] ) and len(text)>0 and text[-1]!=' ' and c!=' ':
                    text.append(' ')
            text.append(c)
            if (c=='/' or c=='*' or c=='-' or c=='=') and text[-1]!=' ':
                text.append(' ')
            
        text="".join(text)
        tokens=nltk.word_tokenize(text)
        lb=[0]*len(tokens)
        for opin in opins:
            opin_cnt[opin[3]]+=1
            token_idx, pt, tag_on=0, 0, False
            for ix, c in enumerate(sent.find('text').text):
                if pt>=len(tokens[token_idx]):
                    pt=0
                    token_idx+=1
                    if token_idx>=len(tokens):
                        break
                if ix==opin[1]: #from
                    assert pt==0 and c!=' '
                    lb[token_idx]=1
                    tag_on=True
                elif ix==opin[2]: #to
                    assert pt==0
                    tag_on=False   
                elif tag_on and pt==0 and c!=' ':
                    lb[token_idx]=2
                if c==' ' or ord(c)==160:
                    pass
                elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                    pt+=2
                else:
                    pt+=1
        if len(opins) > 0:
            id_list.append(sent.attrib['id'])
            sentence_list.append(text)
            term_list.append(lb)

    clean_id_list, text_list, text_list_split, pos_list, senti_list, label_list = process_text(id_list, sentence_list,
                                                                                               term_list,
                                                                                               sentinet,
                                                                                               gloss_embedding,
                                                                                               gloss_embedding_norm,
                                                                                               'discrete')

    for idx in range(len(text_list)):
        assert len(text_list_split[idx]) == len(label_list[idx])
        corpus.append({"id": clean_id_list[idx], "sentence": text_list[idx], "labels": label_list[idx],
                        "pos_sent": pos_list[idx], "senti_sent": senti_list[idx], "sentence_split": text_list_split[idx]})

    print('number of sentences with each label: ', opin_cnt)
    print('number of training samples: ',len(corpus))
    print('number of total sentences: ', sent_cnt)
    print('number of total aspects: ', aspect_cnt)
    return corpus


train_corpus = parse_SemEval14('../raw_data/aspect_data/semeval14/train/Laptop_Train_v2.xml')

with open("../preprocessed_data_ae/laptop/train_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:-valid_split]}, fw, sort_keys=True, indent=4)
with open("../preprocessed_data_ae/laptop/dev_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[-valid_split:]}, fw, sort_keys=True, indent=4)

test_corpus = parse_SemEval14('../raw_data/aspect_data/semeval14/test/Laptops_Test_Gold.xml')

with open("../preprocessed_data_ae/laptop/test_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)


train_corpus = parse_SemEval14('../raw_data/aspect_data/semeval14/train/Restaurants_Train_v2.xml')

with open("../preprocessed_data_ae/restaurant/train_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:-valid_split]}, fw, sort_keys=True, indent=4)
with open("../preprocessed_data_ae/restaurant/dev_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[-valid_split:]}, fw, sort_keys=True, indent=4)

test_corpus = parse_SemEval14('../raw_data/aspect_data/semeval14/test/Restaurants_Test_Gold.xml')

with open("../preprocessed_data_ae/restaurant/test_newpos.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)
