# SentiLARE: Sentiment-Aware Language Representation Learning with Linguistic Knowledge

## Introduction

SentiLARE is a sentiment-aware pre-trained language model enhanced by linguistic knowledge. You can read our [paper](https://www.aclweb.org/anthology/2020.emnlp-main.567/) for more details. This project is a PyTorch implementation of our work.

## Dependencies

* Python 3
* NumPy
* Scikit-learn
* PyTorch >= 1.3.0
* PyTorch-Transformers (Huggingface) 1.2.0
* TensorboardX
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) 0.2.6 (Optional, used for linguistic knowledge acquisition during pre-training and fine-tuning)
* NLTK (Optional, used for linguistic knowledge acquisition during pre-training and fine-tuning)

## Quick Start for Fine-tuning

### Datasets of Downstream Tasks

Our experiments contain sentence-level sentiment classification (e.g. SST / MR / IMDB / Yelp-2 / Yelp-5) and aspect-level sentiment analysis (e.g. Lap14 / Res14 / Res16). You can download the pre-processed datasets ([Google Drive](https://drive.google.com/drive/folders/1v84riTNxCMJi3HWhJdDNyBryCtTTfNjy?usp=sharing) / [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/f6baaff5c398463388b2/)) of the downstream tasks. The detailed description of the data formats is attached to the datasets.

### Fine-tuning

To quickly conduct the fine-tuning experiments, you can directly download the checkpoint ([Google Drive](https://drive.google.com/drive/folders/1v84riTNxCMJi3HWhJdDNyBryCtTTfNjy?usp=sharing) / [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/f6baaff5c398463388b2/)) of our pre-trained model. We show the example of fine-tuning SentiLARE on SST as follows:

```shell
cd finetune
CUDA_VISIBLE_DEVICES=0,1,2 python run_sent_sentilr_roberta.py \
          --data_dir data/sent/sst \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name sst \
          --do_train \
          --do_eval \
          --max_seq_length 256 \
          --per_gpu_train_batch_size 4 \
          --learning_rate 2e-5 \
          --num_train_epochs 3 \
          --output_dir sent_finetune/sst \
          --logging_steps 100 \
          --save_steps 100 \
          --warmup_steps 100 \
          --eval_all_checkpoints \
          --overwrite_output_dir
```

Note that `data_dir` is set to the directory of pre-processed SST dataset, and `model_name_or_path` is set to the directory of the pre-trained model checkpoint. `output_dir` is the directory to save the fine-tuning checkpoints. You can refer to the fine-tuning codes to get the description of other hyper-parameters.

More details about fine-tuning SentiLARE on other datasets can be found in [`finetune/README.MD`](https://github.com/thu-coai/SentiLARE/tree/master/finetune).

### POS Tagging and Polarity Acquisition for Downstream Tasks

During pre-processing, we tokenize the original datasets with NLTK, tag the sentences with Stanford Log-Linear Part-of-Speech Tagger, and obtain the sentiment polarity with Sentence-BERT. We further release the original datasets and the pre-processing scripts (to be announced soon), so you can follow our pipeline to acquire linguistic knowledge for your own datasets.

## Pre-training

If you want to conduct pre-training by yourself instead of directly using the checkpoint we provide, this part may help you pre-process the pre-training dataset and run the pre-training scripts.

### Dataset

We use Yelp Dataset Challenge 2019 as our pre-training dataset. According to the [Term of Use](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf) of Yelp dataset, you should download [Yelp dataset](https://www.yelp.com/dataset) on your own.

### POS Tagging and Polarity Acquisition for Pre-training Dataset

Similar to fine-tuning, we also conduct part-of-speech tagging and sentiment polarity acquisition on the pre-training dataset. The pre-processing scripts will be announced soon. Note that since the pre-training dataset is quite large, the pre-processing procedure may take a long time because we need to use Sentence-BERT to obtain the representation vectors of all the sentences in the pre-training dataset.

### Pre-training

The pre-training codes will be released soon.

## Citation

```
@inproceedings{ke-etal-2020-sentilare,
    title = "{S}enti{LARE}: Sentiment-Aware Language Representation Learning with Linguistic Knowledge",
    author = "Ke, Pei  and Ji, Haozhe  and Liu, Siyang  and Zhu, Xiaoyan  and Huang, Minlie",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    pages = "6975--6988",
}
```

**Please kindly cite our paper if this paper and the codes are helpful.**

## Thanks

Many thanks to the GitHub repositories of [Transformers](https://github.com/huggingface/transformers) and [BERT-PT](https://github.com/howardhsu/BERT-for-RRC-ABSA). Part of our codes are modified based on their codes.
