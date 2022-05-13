import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification

#추론 코드

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    #추론을 위한 batch_size이므로 좀 더 커도 됨
    p.add_argument('--batch_size', type=int, default=256)
    #하나만 나오는 것
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config

def read_text():
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip()]

    return lines

def main(config):
    #save된 모델을 불러옴
    saved_data = torch.load(
        config.model_fn,
        #torch에서 load되면, 자동으로 이전 device로 올라감
        #이것을 방지하는 코드
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    #처음부터 입력이 다 들어오고 나서 실행되도록 코드를 짬
    #for문에 넣어서 짜도 괜찮음
    lines = read_text()

    with torch.no_grad():
        #다운로드가 되어있다면 다시 받지 않음
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)
        model_loader = AlbertForSequenceClassification if train_config.use_albert else BertForSequenceClassification

        model = model_loader.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )

        #fine tuning한 weight parameter를 불러옴
        model.load_state_dict(bert_best)

        #gpu로 넘김
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        #모델의 디바이스는 없는 상태라서
        #모델의 첫번째 파라미터의 device를 보면, 모델이 어떤 device에 올라가있는지 알 수 있음
        device = next(model.parameter()).device

        #까먹으면 안됨, 성능이 떨어짐
        model.eval()

        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(
                lines[idx:idx + config.batch_size],
                padding=True,
                truncation=True, #max_length를 기준으로 잘라줌
                return_tensors='pt', #파이토치 텐서
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            y_hat = F.softmax(
                model(x, attention_mask=mask).logits, #(n, |c|)
                dim=-1 #가장 끝의 차원을 대상으로 softmax
                )

            y_hats += [y_hat]

        y_hats = torch.cat(y_hats, dim=0)
        #|y_hats| = (len(lines), n_classes) = (n * minibatch, |c|)

        probs, indice = y_hats.cpu().topk(config.top_k)
        # |indices| = (len(lines), top_k)

        for i in range(len(lines)):
            sys.stdout.write('%s\t%s\n' % (
                ' '.join(
                    [
                        index_to_label[ int( indice[i][j] ) ] for j in range(config.top_k)
                    ]), 
                lines[i]
            ))

if __name__ == '__main__':
    config = define_argparser()
    main(config)
