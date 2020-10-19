# Process the input data in aspect-based tasks
# The code is modified based on absa_data_utils.py in BERT-for-RRC-ABSA.
# https://github.com/howardhsu/BERT-for-RRC-ABSA/blob/master/pytorch-pretrained-bert/src/absa_data_utils.py


import json
import os
from collections import defaultdict
import random

from pytorch_transformers import RobertaTokenizer

class ABSATokenizer_roberta(RobertaTokenizer):
    # For aspect term extraction task
    def subword_tokenize(self, tokens, labels, pos, senti):
        split_tokens, split_labels, split_pos, split_senti= [], [], [], []
        for ix, token in enumerate(tokens):
            sub_tokens=self._tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if labels[ix]==1 and jx>0:
                    split_labels.append(2)
                else:
                    split_labels.append(labels[ix])
                split_pos.append(pos[ix])
                split_senti.append(senti[ix])
        return split_tokens, split_labels, split_pos, split_senti


class InputExample(object):
    """A single training/test example for aspect-based tasks."""

    def __init__(self, guid, text_a, text_a_split, text_a_pos, text_a_senti, label,
                 text_b = None, text_b_split = None, text_b_pos = None, text_b_senti = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_a_split: List[string]. The tokenized result of the first sequence.
            text_a_pos: List[int]. The POS tag of each word in the first sequence.
            text_a_senti: List[int]. The sentiment polarity of each word in the first sequence.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_b_split: List[string]. The tokenized result of the second sequence.
            text_b_pos: List[int]. The POS tag of each word in the second sequence.
            text_b_senti: List[int]. The sentiment polarity of each word in the second sequence.
            label: (Optional) string. The label of the example.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_a_split = text_a_split
        self.text_a_pos = text_a_pos
        self.text_a_senti = text_a_senti
        self.text_b = text_b
        self.text_b_split = text_b_split
        self.text_b_pos = text_b_pos
        self.text_b_senti = text_b_senti
        self.label = label



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, pos_ids, senti_ids, polarity_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        self.senti_ids = senti_ids
        self.polarity_ids = polarity_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
        
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

        
        
class AeProcessor(DataProcessor):
    """Processor for the SemEval Aspect Term Extraction ."""

    def get_train_examples(self, data_dir, fn="train_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")
    
    def get_test_examples(self, data_dir, fn="test_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class. 0, 1, 2 mean O, B, I, respectively."""
        return [0, 1, 2]

    def _create_examples(self, lines, set_type):
        """Creates examples for the train / dev / test sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = " ".join(lines[ids]['sentence_split'])
            text_a_split = lines[ids]['sentence_split']
            text_a_pos = lines[ids]['pos_sent']
            text_a_senti = lines[ids]['senti_sent']
            label = lines[ids]['labels']
            examples.append(
                InputExample(guid=guid, text_a = text_a, text_a_split=text_a_split, text_a_pos=text_a_pos, text_a_senti=text_a_senti, label=label))
        return examples


class AcdProcessor_14(DataProcessor):
    """Processor for the SemEval Aspect Category Detection (Res14)."""

    def get_train_examples(self, data_dir, fn="train_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ['food', 'service', 'price', 'ambience', 'anecdotes/miscellaneous']

    def _create_examples(self, lines, set_type):
        """Creates examples for the train / dev / test sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids)
            text_a = " ".join(lines[ids]['sentence_split'])
            text_a_split = lines[ids]['sentence_split']
            text_a_pos = lines[ids]['pos_sent']
            text_a_senti = lines[ids]['senti_sent']
            label = lines[ids]['term']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_a_split=text_a_split, text_a_pos=text_a_pos,
                             text_a_senti=text_a_senti, label=label))
        return examples


class AcdProcessor_16(DataProcessor):
    """Processor for the SemEval Aspect Category Detection (Res16)."""

    def get_train_examples(self, data_dir, fn="train_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ['RESTAURANT#GENERAL', 'RESTAURANT#PRICES', 'RESTAURANT#MISCELLANEOUS',
                'FOOD#PRICES', "FOOD#QUALITY", "FOOD#STYLE_OPTIONS", "DRINKS#PRICES",
                "DRINKS#QUALITY", "DRINKS#STYLE_OPTIONS", "AMBIENCE#GENERAL", "SERVICE#GENERAL",
                "LOCATION#GENERAL"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the train / dev / test sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids)
            text_a = " ".join(lines[ids]['sentence_split'])
            text_a_split = lines[ids]['sentence_split']
            text_a_pos = lines[ids]['pos_sent']
            text_a_senti = lines[ids]['senti_sent']
            label = lines[ids]['term']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_a_split=text_a_split, text_a_pos=text_a_pos,
                             text_a_senti=text_a_senti, label=label))
        return examples

class AscProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_train_examples(self, data_dir, fn="train_cased_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev_cased_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")
    
    def get_test_examples(self, data_dir, fn="test_cased_newpos.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ["positive", "negative", "neutral"]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the train / dev / test sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['term']
            text_a_split = lines[ids]['term_split']
            text_a_pos = lines[ids]['pos_term']
            text_a_senti = lines[ids]['senti_term']
            text_b = lines[ids]['sentence']
            text_b_split = lines[ids]['sentence_split']
            text_b_pos = lines[ids]['pos_sent']
            text_b_senti = lines[ids]['senti_sent']
            label = lines[ids]['polarity']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_a_split=text_a_split, text_a_pos=text_a_pos, text_a_senti=text_a_senti,
                             text_b=text_b, text_b_split=text_b_split, text_b_pos=text_b_pos, text_b_senti=text_b_senti, label=label))
        return examples


def convert_examples_to_features_roberta(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 mode=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        if mode != "ae":
            tokens_a, pos_a, senti_a = [], [], []
            for i, tok in enumerate(example.text_a_split):
                tok_list = tokenizer.tokenize(tok)
                tokens_a.extend(tok_list)
                pos_a.extend([example.text_a_pos[i]] * len(tok_list))
                senti_a.extend([example.text_a_senti[i]] * len(tok_list))
        else:
            tokens_a, labels_a, pos_a, senti_a, = tokenizer.subword_tokenize(
                [token for token in example.text_a_split], example.label,
                example.text_a_pos, example.text_a_senti)

        if example.text_b:
            if mode != "ae":
                tokens_b, pos_b, senti_b = [], [], []
                for i, tok in enumerate(example.text_b_split):
                    tok_list = tokenizer.tokenize(tok)
                    tokens_b.extend(tok_list)
                    pos_b.extend([example.text_b_pos[i]] * len(tok_list))
                    senti_b.extend([example.text_b_senti[i]] * len(tok_list))
            else:
                tokens_b, labels_b, pos_b, senti_b, = tokenizer.subword_tokenize(
                    [token for token in example.text_b_split], example.label,
                    example.text_b_pos, example.text_b_senti)
        else:
            tokens_b, pos_b, senti_b = None, None, None

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, pos_a, senti_a, pos_b, senti_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]
                pos_a = pos_a[:(max_seq_length - special_tokens_count)]
                senti_a = senti_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        # 4 in POS tags means others, 2 in word-level polarity labels means neutral, and
        # 5 in sentence-level sentiment labels means unknown sentiment
        tokens = tokens_a + [sep_token]
        pos_ids = pos_a + [4]
        senti_ids = senti_a + [2]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            pos_ids += [4]
            senti_ids += [2]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            pos_ids += pos_b + [4]
            senti_ids += senti_b + [2]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            pos_ids = pos_ids + [4]
            senti_ids = senti_ids + [2]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            pos_ids = [4] + pos_ids
            senti_ids = [2] + senti_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            pos_ids = ([4] * padding_length) + pos_ids
            senti_ids = ([2] * padding_length) + senti_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            pos_ids =  pos_ids + ([4] * padding_length)
            senti_ids = senti_ids + ([2] * padding_length)

        # During fine-tuning, the sentence-level label is set to unknown
        polarity_ids = [5] * max_seq_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(pos_ids) == max_seq_length
        assert len(senti_ids) == max_seq_length
        assert len(polarity_ids) == max_seq_length


        if output_mode == "classification":
            if mode != "ae":
                if mode != "acd14" and mode != "acd16":
                    # label assignment for ATSC and ACSC
                    label_id = label_map[example.label]
                else:
                    # label assignment for ACD
                    label_id = [0] * len(label_map)
                    for idx in example.label:
                        label_id[label_map[idx]] = 1
            else:
                # label assignment for ATE
                label_id = [-1] * len(input_ids)  # -1 is the index to ignore
                # truncate the label length if it exceeds the limit.
                lb = [label_map[label] for label in labels_a]
                if len(lb) > max_seq_length - special_tokens_count:
                    lb = lb[0:(max_seq_length - special_tokens_count)]
                if pad_on_left:
                    label_id[max_seq_length - special_tokens_count - len(lb):max_seq_length - special_tokens_count] = lb  # except [CLS] and [SEP]
                else:
                    label_id[1:len(lb) + 1] = lb  # except [CLS]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # Print a few data samples to check the data format
        if ex_index <= 5:
            print(tokens)
            print(input_ids)
            print(input_mask)
            print(segment_ids)
            print(pos_ids)
            print(senti_ids)
            print(polarity_ids)
            print(label_id)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              pos_ids=pos_ids,
                              senti_ids=senti_ids,
                              polarity_ids=polarity_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, pos_a, senti_a, pos_b, senti_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            pos_a.pop()
            senti_a.pop()
        else:
            tokens_b.pop()
            pos_b.pop()
            senti_b.pop()