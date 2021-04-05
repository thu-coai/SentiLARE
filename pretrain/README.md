# Pre-training of SentiLARE 

## Input Files

Similar to fine-tuning, we need to first pre-process the pre-training dataset to acquire the tokenized texts, POS tags, word-level sentiment polarities, and the review-level sentiment labels. Here, we give an example of input files. You can simply modify the [pre-processing codes](https://github.com/thu-coai/SentiLARE/tree/master/preprocess) for fine-tuning to construct these files for pre-training.

1) Raw file

```
{"stars":1.0, "text":"Total bill for this horrible service? Over $8Gs. These crooks actually had the nerve to charge us $69 for 3 pills. I checked online the pills can be had for 19 cents EACH! Avoid Hospital ERs at all costs."}
```

2) yelp_bert_format_word_and_pos.txt in data_label.py (including tokenized texts and the POS tags of each word)

```
Total#a bill#n for#u this#u horrible#a service#n ?#u
Over#u $#u 8Gs#u .#u
These#u crooks#n actually#r had#v the#u nerve#n to#u charge#v us#u $#u 69#u for#u 3#u pills#n .#u
I#u checked#v online#n the#u pills#n can#u be#v had#v for#u 19#u cents#n EACH#n !#u
Avoid#v Hospital#n ERs#n at#u all#u costs#n .#u
```

a / n / v / r / u denotes adjective / noun / verb / adverb / others, respectively.

3) yelp_sentiment_label.txt in data_label.py (including the word-level sentiment polarities)

```
2 2 2 2 0 2 2
2 2 2 2
2 0 1 0 2 2 2 1 2 2 2 2 2 0 2
2 1 2 2 0 2 1 0 2 2 2 2 2
0 1 1 2 2 2 2
```

0 / 1 / 2 denotes negative / positive / neutral, respectively. For example, in the first line, only the polarity of "horrible" is negative while others are neutral. 

4) yelp_stars.txt in data_label.py (including the review-level sentiment label)

```
1.0
```

## Pre-training

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python post_pretrain.py
```

