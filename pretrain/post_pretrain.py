import os
import json
import logging
import argparse
import random
import numpy as np
from tqdm import trange, tqdm
import subprocess

import torch
from torch.utils.data import RandomSampler, DataLoader

from tokenization_roberta import RobertaTokenizer
from modeling_roberta import RobertaForPreTraining
from optimization import AdamW, WarmupLinearSchedule
from data_label import Yelp
from utils import set_log, AverageMeter

logger = logging.getLogger()

MAX_SEQ_LEN = 128
BATCH_SIZE = 400
EPOCHS = 1
WARMUP = 0.1
LR = 5e-5
WORKERS = 5

MODEL_DIR = './pretrain_model'
PRETRAIN_BERT = 'roberta-base'
DEBUG = False

is_pos_embedding = True
is_senti_embedding = True
is_polarity_embedding = True


def set_args(parser):
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--seed', type=int, default=1023)

    parser.add_argument('--model_name', type=str, default='senti-roberta')
    parser.add_argument('--task_ratio', type=float, default=0.2, help='Specify the ratio of two tasks. Sentence polarity supervision: x, Conditional masked LM: 1-x')
    parser.add_argument('--no_mlm', action='store_true', help='set true for no masked language modeling')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup', type=float, default=WARMUP)
    parser.add_argument('--learning_rate', type=float, default=LR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--max_seq_length', type=int, default=MAX_SEQ_LEN)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--fp16', action='store_true')

def specify_path(args):
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    # Specify model name
    args.model_name = args.model_name + \
                        '-lr-{}-'.format(LR) + \
                        'msl-{}-'.format(MAX_SEQ_LEN) + \
                        'bs-{}-'.format(BATCH_SIZE) + \
                        'ep-{}-'.format(EPOCHS) + \
                        'wp-{}-'.format(WARMUP) + \
                        'ratio-{}'.format(args.task_ratio) + \
                        '-discrete'

    # Specify the save directory
    args.save_dir = os.path.join(MODEL_DIR, args.model_name)
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    args.log_dir = os.path.join(args.save_dir, 'log.txt')
    version = 1
    while os.path.isfile(args.log_dir):
        args.log_dir = args.log_dir.split('log')[0] + 'log{}.txt'.format(version)
        version += 1
    
    args.pretrain_bert_dir = os.path.join(MODEL_DIR, PRETRAIN_BERT)
    
    
def specify_device(args):
    # Set the device
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.batch_size = BATCH_SIZE // args.grad_accum_steps
    args.n_gpu = torch.cuda.device_count()


def specify_seed(args):
    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)


def init_optimizer(args, model, optimization_step):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=int(optimization_step * args.warmup), t_total=optimization_step
    )

    return optimizer, scheduler


def train(args, model, tokenizer, loader, optimizer, scheduler):
    # Pre-train
    global_step = 0
    model.train()
    meters = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
    for cur_epoch in trange(int(args.epochs), desc="Epoch"):
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = tuple(t.to(args.device) for t in batch)
            masked_lm_loss, pos_tag_loss, senti_loss, polarity_loss = model(*batch)
            loss = masked_lm_loss + pos_tag_loss + senti_loss + polarity_loss
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
                masked_lm_loss = masked_lm_loss.mean()
                pos_tag_loss = pos_tag_loss.mean()
                senti_loss = senti_loss.mean()
                polarity_loss = polarity_loss.mean()
            if args.grad_accum_steps > 1:
                loss = loss / args.grad_accum_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            meters[0].update(masked_lm_loss.item())
            meters[1].update(pos_tag_loss.item())
            meters[2].update(senti_loss.item())
            meters[3].update(polarity_loss.item())

            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1
            if step % 10 == 0:
                logger.info('Epoch {} | step {} | Total loss {:.4f} | MLM loss {:.4f} | PTG loss {:.4f} | ST loss {:.4f} | POL loss {:.4f}'.format(cur_epoch, step, sum([m.avg for m in meters]), meters[0].avg, meters[1].avg, meters[2].avg, meters[3].avg))
            
            if (step + 1) % args.grad_accum_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        save(args, model, tokenizer, optimizer, scheduler)
    
        logger.info("saved model at epoch {}".format(cur_epoch))


def save(args, model, tokenizer, optimizer, scheduler):
    # Save model checkpoint
    logger.info("** ** * Saving fine - tuned model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model
    
    model_to_save.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    torch.save(args, os.path.join(args.save_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", args.save_dir)
    
    torch.save(optimizer.state_dict(), os.path.join(args.save_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(args.save_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", args.save_dir)
    
    
def main():
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    
    specify_path(args)
    specify_device(args)
    specify_seed(args)

    print_args = {k: v for k, v in vars(args).items() if k != 'device'}
    print_args = argparse.Namespace(**print_args)
    logger.info('CONFIG:\n%s' %json.dumps(vars(print_args), indent=4, sort_keys=True))

    if not args.no_log:
        set_log(args.log_dir)

    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_bert_dir)
    model = RobertaForPreTraining.from_pretrained(args.pretrain_bert_dir, pos_tag_embedding=is_pos_embedding,
                                                  senti_embedding=is_senti_embedding, polarity_embedding=is_polarity_embedding)
    if args.fp16:
        model.half()
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load pre-training data
    logging.info('Loading yelp pretraining data...')
    yelp = Yelp(args, tokenizer, max_seq_length=args.max_seq_length)
    sampler = RandomSampler(yelp)
    loader = DataLoader(yelp, 
                        sampler=sampler, 
                        batch_size=args.batch_size, 
                        num_workers=WORKERS,
                        pin_memory=args.cuda)
    

    optimization_step = int(len(yelp) / args.batch_size / args.grad_accum_steps) * args.epochs
    optimizer, scheduler = init_optimizer(args, model, optimization_step)
    
    train(args, model, tokenizer, loader, optimizer, scheduler)

if __name__ == '__main__':
    main()
