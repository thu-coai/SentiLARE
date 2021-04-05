# Fine-tuning of SentiLARE 

## Preprocessing

During fine-tuning, we need to first pre-process the raw datasets of downstream tasks, including POS tagging and word-level polarity acquisition. Refer to [`preprocess/README.md`](https://github.com/thu-coai/SentiLARE/tree/master/preprocess) for more implementation details. If you use the pre-processed datasets we provide, you can skip this step.

## Fine-tuning

**Note**: The performance of pre-trained language models may vary with different random seeds, especially on small datasets. The results reported in our paper are the mean values over 5 random seeds.

### Sentence-level Sentiment Classification

SST

```shell
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
          --overwrite_output_dir \
          --seed 42
```

MR

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python run_sent_sentilr_roberta.py \
          --data_dir data/sent/mr \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name mr \
          --do_train \
          --do_eval \
          --max_seq_length 256 \
          --per_gpu_train_batch_size 8 \
          --learning_rate 3e-5 \
          --num_train_epochs 4 \
          --output_dir sent_finetune/mr \
          --logging_steps 50 \
          --save_steps 50 \
          --warmup_steps 20 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --seed 42
```

IMDB

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python run_sent_sentilr_roberta.py \
          --data_dir data/sent/imdb \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name imdb \
          --do_train \
          --do_eval \
          --max_seq_length 512 \
          --per_gpu_train_batch_size 8 \
          --learning_rate 2e-5 \
          --num_train_epochs 3 \
          --output_dir sent_finetune/imdb \
          --logging_steps 150 \
          --save_steps 150 \
          --warmup_steps 100 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --seed 42
```

Yelp-2

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python run_sent_sentilr_roberta.py \
          --data_dir data/sent/yelp2 \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name yelp2 \
          --do_train \
          --do_eval \
          --max_seq_length 512 \
          --per_gpu_train_batch_size 4 \
          --learning_rate 2e-5 \
          --num_train_epochs 3 \
          --output_dir sent_finetune/yelp2 \
          --logging_steps 6000 \
          --save_steps 6000 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --warmup_steps 12600 \
          --seed 42
```

Yelp-5

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python run_sent_sentilr_roberta.py \
          --data_dir data/sent/yelp5 \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name yelp5 \
          --do_train \
          --do_eval \
          --max_seq_length 512 \
          --per_gpu_train_batch_size 4 \
          --learning_rate 2e-5 \
          --num_train_epochs 3 \
          --output_dir sent_finetune/yelp5 \
          --logging_steps 7200 \
          --save_steps 7200 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --warmup_steps 8500 \
          --seed 42
```

### Aspect Term Extraction

Lap14

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python run_ae_sentilr_roberta.py \
        --data_dir data/aspect/ae/lap14  \
        --model_type roberta \
        --model_name_or_path pretrain_model/ \
        --task_name ae \
        --output_dir ae_finetune/lap14 \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --per_gpu_train_batch_size 4 \
        --learning_rate 3e-5 \
        --num_train_epochs 4 \
        --logging_steps 30 \
        --save_steps 30 \
        --eval_all_checkpoints \
        --overwrite_output_dir \
        --seed 3
```

Res14

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python run_ae_sentilr_roberta.py \
        --data_dir data/aspect/ae/res14 \
        --model_type roberta \
        --model_name_or_path pretrain_model/ \
        --task_name ae \
        --output_dir ae_finetune/res14 \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --per_gpu_train_batch_size 4 \
        --learning_rate 3e-5 \
        --num_train_epochs 4 \
        --logging_steps 80 \
        --save_steps 80 \
        --eval_all_checkpoints \
        --overwrite_output_dir \
        --seed 3
```

### Aspect Term Sentiment Classification

Lap14

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_asc_sentilr_roberta.py \
          --data_dir data/aspect/atsc/lap14 \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name asc \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 4 \
          --learning_rate 3e-5 \
          --num_train_epochs 8 \
          --output_dir atsc_finetune/lap14 \
          --logging_steps 50 \
          --save_steps 50 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --seed 3
```

Res14

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_asc_sentilr_roberta.py \
          --data_dir data/aspect/atsc/res14 \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name asc \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 4 \
          --learning_rate 3e-5 \
          --num_train_epochs 8 \
          --output_dir atsc_finetune/res14/ \
          --logging_steps 50 \
          --save_steps 50 \
          --warmup_steps 150 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --seed 3
```

### Aspect Category Detection

Res14

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_acd_sentilr_roberta.py \
        --data_dir data/aspect/acd/res14 \
        --model_type roberta \
        --model_name_or_path pretrain_model/ \
        --task_name acd14 \
        --output_dir acd_finetune/res14 \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --per_gpu_train_batch_size 4 \
        --learning_rate 3e-5 \
        --num_train_epochs 4 \
        --logging_steps 30 \
        --save_steps 30 \
        --eval_all_checkpoints \
        --overwrite_output_dir \
        --warmup_steps 100 \
        --seed 3 \
```

Res16

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_acd_sentilr_roberta.py \
        --data_dir data/aspect/acd/res16 \
        --model_type roberta \
        --model_name_or_path pretrain_model/ \
        --task_name acd16 \
        --output_dir acd_finetune/res16 \
        --max_seq_length 128 \
        --do_train \
        --do_eval \
        --per_gpu_train_batch_size 4 \
        --learning_rate 3e-5 \
        --num_train_epochs 6 \
        --logging_steps 80 \
        --save_steps 80 \
        --eval_all_checkpoints \
        --overwrite_output_dir \
        --seed 3
```

### Aspect Category Sentiment Classification

Res14

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_asc_sentilr_roberta.py \
          --data_dir data/aspect/acsc/res14 \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name asc \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 4 \
          --learning_rate 3e-5 \
          --num_train_epochs 8 \
          --output_dir acsc_finetune/res14 \
          --logging_steps 100 \
          --save_steps 100 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --seed 3
```

Res16

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_asc_sentilr_roberta.py \
          --data_dir data/aspect/acsc/res16 \
          --model_type roberta \
          --model_name_or_path pretrain_model/ \
          --task_name asc \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 8 \
          --learning_rate 3e-5 \
          --num_train_epochs 8 \
          --output_dir acsc_finetune/res16 \
          --logging_steps 40 \
          --save_steps 40 \
          --warmup_steps 60 \
          --eval_all_checkpoints \
          --overwrite_output_dir \
          --seed 3
```

