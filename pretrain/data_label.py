#-*- coding: utf-8 -*-
import os
import json
import logging
import copy
from tqdm import tqdm, trange
import re 
import random
import numpy as np
from multiprocessing import Pool
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
import time
import codecs


from utils import set_log
from tokenization_roberta import RobertaTokenizer

logger = logging.getLogger()

YELP_POS_DIR = './yelp_bert_format_word_and_pos.txt'
YELP_STAR_DIR = './yelp_stars.txt'
SENTIWORD_DIR = './yelp_sentiment_label.txt'


class Yelp(Dataset):
    def __init__(self, args, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.pattern = r'([.?!])'
        self.max_seq_length = max_seq_length
        self.pos_tag_ids_map = {'#v':0, '#a':1, '#r':2, '#n':3, '#u':4}
        self.load_sentiscore()
        self.load_pos()
        self.args = args

    # Load the word-level sentiment polarity
    def load_sentiscore(self, filename=SENTIWORD_DIR):
        self.sentiscores_total = []
        self.sentiwords_list = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                self.sentiscores_total.append(line.strip())
        print('load sentiment scores complete')

    # Load the POS tags
    def load_pos(self, filename=YELP_POS_DIR, starfilename = YELP_STAR_DIR):
        self.exs = []
        self.exs_senti = []
        senti_cnt = 0
        with open(filename, 'r') as f:
            para = []
            para_senti = []
            for line in f.readlines():
                if line.strip() != '':
                    para.append(line.strip())
                    para_senti.append(self.sentiscores_total[senti_cnt])
                    senti_cnt += 1
                else:
                    self.exs.append(para)
                    self.exs_senti.append(para_senti)
                    para = []
                    para_senti = []
                    
        count = 0
        print(len(self.exs))
        print(senti_cnt)
        with open(starfilename, 'r') as f:
            for i, line in enumerate(f.readlines()):
                self.exs[i].append(eval(line) - 1.0)
                count += 1
        print('load yelp star complete')
        assert count == len(self.exs)

        print('load pos tags complete')

    def __len__(self):
        return len(self.exs)
    
    def get_pos_tag_ids(self, text):
        # acquire the pos tag
        ids = []
        for c in text.split():
            ids.append(self.pos_tag_ids_map[c[-2:]])
        return ids

    def get_senti_ids(self, text, senti):
        # acquire the word-level polarity
        return [int(label) for label in senti.split()]

    def get_clean_text(self, text):
        # remove the pos tag to get texts
        clean_text = []
        for c in text.split():
            clean_text.append(c[:-2])
        return clean_text
    
    def get_ids(self, text, senti):
        # get token id, pos id and polarity id
        clean_text = self.get_clean_text(text)
        p_ids = self.get_pos_tag_ids(text)
        s_ids = self.get_senti_ids(text, senti)

        assert len(clean_text) == len(p_ids)
        assert len(clean_text) == len(s_ids)

        return clean_text, p_ids, s_ids
        
    def vectorize(self, text, ids):
        # convert tokens to ids
        toks = []
        toks_ids = []
        toks_text = []
        for i, c in enumerate(text):
            if type(c) != list:
                t = self.tokenizer.tokenize(c)
                toks_text.extend(t)
                t = self.tokenizer.convert_tokens_to_ids(t)
            elif type(c[0]) == int:
                t = c
                toks_text.extend([self.tokenizer.mask_token] * len(t))
            else:
                toks_text.extend(c)
                t = self.tokenizer.convert_tokens_to_ids(c)
            toks.extend(t)
            if type(ids[i]) == list:
                toks_ids.extend(self.tokenizer.convert_tokens_to_ids(ids[i]))
            elif type(ids[i]) == str:
                _t = self.tokenizer.tokenize(ids[i])
                toks_ids.extend(self.tokenizer.convert_tokens_to_ids(_t))
            else:
                toks_ids.extend([ids[i]] * len(t))
        assert len(toks) == len(toks_ids)
        assert len(toks) == len(toks_text)
        
        return toks, toks_ids,toks_text

    def __getitem__(self, idx):
        sents = self.exs[idx]
        senti_ids = self.exs_senti[idx]
        
        input_seg = " ".join(sents[:-1])
        senti_seg = " ".join(senti_ids)

        rating = sents[-1]
        # Roberta has no nsp objective
        nsp_label = -1
        
        input_triple = self.get_ids(input_seg, senti_seg)
        
        input_triple, input_labels = self.random_whole_word(input_triple)
        
        _input_text, _input_ids_list = input_triple[0], input_triple[1:] + input_labels
        
        input_ids_list = []
        
        for input_ids in _input_ids_list:
            
            input_text, input_ids, input_text_backup = self.vectorize(_input_text, input_ids)
            
            input_ids_list.append(input_ids)
        
        _truncate_seq(input_text, self.max_seq_length - 2)
        for s in input_ids_list:
            _truncate_seq(s, self.max_seq_length - 2)
        _truncate_seq(input_text_backup, self.max_seq_length - 2)
        
        p_ids = [4] + input_ids_list[0] + [4]
        s_ids = [2] + input_ids_list[1] + [2]
        
        lm_label = [-1] + input_ids_list[2] + [-1]
        p_label = [-1] + input_ids_list[3] + [-1]
        s_label = [-1] + input_ids_list[4] + [-1]
        
        prob = random.random()
        if prob < self.args.task_ratio:
            # Late Supervision
            polarity_ids = [5] * len(p_ids)
            polarity_label = [int(rating)] + [-1] * (len(p_ids) - 1)
        else:
            # Early Fusion
            polarity_ids = [5] + [int(rating)] * len(input_text) + [5]
            polarity_label = [-1] * len(p_ids)
    
        tokens = []
        segment_ids = []

        tokens.append(self.tokenizer.cls_token_id)
        segment_ids.append(0)
        for token in input_text:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(self.tokenizer.sep_token_id)
        segment_ids.append(0)

        input_text_backup = [self.tokenizer.cls_token] + input_text_backup + [self.tokenizer.sep_token]
        
        input_ids = tokens
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.tokenizer.pad_token_id)
            input_text_backup.append(self.tokenizer.pad_token)
            input_mask.append(0)
            segment_ids.append(0)
            s_ids.append(2)
            p_ids.append(4)
            lm_label.append(-1)
            p_label.append(-1)
            s_label.append(-1)
            polarity_ids.append(5)
            polarity_label.append(-1)
        
        assert len(input_ids) == self.max_seq_length, len(input_ids)
        assert len(input_text_backup) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(s_ids) == self.max_seq_length
        assert len(p_ids) == self.max_seq_length
        assert len(lm_label) == self.max_seq_length, len(lm_label)
        assert len(p_label) == self.max_seq_length
        assert len(s_label) == self.max_seq_length
        assert len(polarity_ids) == self.max_seq_length
        assert len(polarity_label) == self.max_seq_length

        return_tensors = (torch.tensor(input_ids), 
                          torch.tensor(input_mask), 
                          torch.tensor(segment_ids), 
                          torch.tensor(lm_label), 
                          torch.tensor(nsp_label),
                          torch.tensor(p_ids),
                          torch.tensor(s_ids),
                          torch.tensor(polarity_ids),
                          torch.tensor(p_label),
                          torch.tensor(s_label),
                          torch.tensor(polarity_label))

        return return_tensors
    
    def random_whole_word(self, vecs):
        words, p_ids, s_ids = vecs
        # words: the tokenized clean text
        # p_ids: pos_tag indices
        # s_ids: senti_word indices
        words_label = []
        p_label = []
        s_label = []
        new_words = []
        for i, word in enumerate(words):
            prob = random.random()
            if s_ids[i] == 2:
                # mask ordinary token with 15% probability
                if prob < 0.15:
                    ori_p = p_ids[i]
                    ori_s = s_ids[i]
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        toks = self.tokenizer.tokenize(word)
                        if toks != []:
                            assert(len(toks) > 0)
                            # mask each subword in this word
                            new_words.append([self.tokenizer.mask_token_id] * len(toks))
                            words_label.append(toks)
                            p_ids[i] = 4
                            s_ids[i] = 2

                    # 10% (0.8~0.9) randomly change token to random token
                    elif prob < 0.9:
                        # random get one word from the example
                        rand_ex = self.exs[random.randint(0, len(self.exs)-1)][0].split()
                        to_replace = self.tokenizer.tokenize(rand_ex[random.randint(0, len(rand_ex)-1)][:-2])
                        toks = self.tokenizer.tokenize(word)
                        if to_replace != [] and toks != []:
                            if len(to_replace) >= len(toks):
                                to_replace = to_replace[:len(toks)]
                            else:
                                to_replace += (len(toks) - len(to_replace)) * [self.tokenizer.mask_token]
                            assert(to_replace != [])
                            new_words.append(to_replace)
                            words_label.append(toks)
                            p_ids[i] = 4
                            s_ids[i] = 2
                    # -> rest 10% randomly keep current token                    
                    else:
                        new_words.append(word)
                        words_label.append(word)
                    if ori_p != 4:  # the pos_tag of ordinary word is not unknown
                        p_label.append(ori_p)
                        s_label.append(ori_s)
                    else:
                        p_label.append(-1)
                        s_label.append(-1)
                else:
                    # no masking token (will be ignored by loss function later)
                    new_words.append(word)
                    words_label.append(-1)
                    p_label.append(-1)
                    s_label.append(-1)
            else:
                # mask senti words with 30% probability
                if prob < 0.3:
                    ori_p = p_ids[i]
                    ori_s = s_ids[i]
                    prob /= 0.3
                    if prob < 0.8:
                        toks = self.tokenizer.tokenize(word)
                        if toks != []:
                            assert(len(toks) > 0), '{}, {}'.format(word, words)
                            new_words.append([self.tokenizer.mask_token_id] * len(toks))
                            words_label.append(toks)
                            p_ids[i] = 4
                            s_ids[i] = 2
                    elif prob < 0.9:
                        # replace with another word
                        rand_ex = self.exs[random.randint(0, len(self.exs)-1)][0].split()
                        to_replace = self.tokenizer.tokenize(rand_ex[random.randint(0, len(rand_ex)-1)][:-2])
                        
                        toks = self.tokenizer.tokenize(word)
                        if to_replace != [] and toks != []:
                            if len(to_replace) >= len(toks):
                                to_replace = to_replace[:len(toks)]
                            else:
                                to_replace += (len(toks) - len(to_replace)) * [self.tokenizer.mask_token]
                            assert(to_replace != [])
                            new_words.append(to_replace)
                            words_label.append(toks)    
                            p_ids[i] = 4
                            s_ids[i] = 2
                    else:
                        new_words.append(word)
                        words_label.append(word)
                    p_label.append(ori_p)
                    s_label.append(ori_s)
                else:
                    new_words.append(word)
                    words_label.append(-1)
                    p_label.append(-1)
                    s_label.append(-1)

        label_vecs = (words_label, p_label, s_label)
        vecs = (new_words, p_ids, s_ids)
        return vecs, label_vecs

        
def _truncate_seq(tokens, max_length):
    while len(tokens) > max_length:
        tokens.pop()
        

def main():
    set_log()
    tokenizer = RobertaTokenizer.from_pretrained('./pretrain_model/roberta-base')
    yelp = Yelp(None, tokenizer)
    for item_id in range(10):
        yelp.__getitem__(item_id)

    
if __name__ == '__main__':
    main()
