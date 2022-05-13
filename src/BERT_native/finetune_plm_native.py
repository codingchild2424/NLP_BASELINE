import argparse
import random
from signal import valid_signals

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from bert_trainer import BertTrainer as Trainer
from bert_data_loader import TextClassificationDataset, TextClassificationCollator
from utils import read_text

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)

    #여기에서 원하는 모델의 이름을 작성하면 됨
    #다른 사람이 huggingface hub에 올린 것
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base') 
    p.add_argument('--use_albert', action='store_true')

    #GPU가 커야 함, 김기현 GPU memory 11GB - 2080ti
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    #GPU 메모리에 따라서 배치를 조절, 김기현 80으로 둠(가급적 2^n으로 설정)
    #GPU가 터지면, 낮추기
    p.add_argument('--batch_size', type=int, default=32)
    #base 기준으로 2번만 돌려도 충분함
    p.add_argument('--n_epochs', type=int, default=5)

    #warmup이 끝났을 때, learning rate
    p.add_argument('--lr', type=float, default=5e-5)
    #자연어 생성 강의에서 이야기했음
    #adam을 쓰면서 성능을 높이는 기법 - warmup
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)

    #warmup을 안하면 rdam을 써야함
    #이때 권장 lr = 1e-4, 그리고 warmup_ratio = 0
    p.add_argument('--use_radam', action='store_true')
    #trainset에서 랜덤하게 추출
    p.add_argument('--valid_ratio', type=float, default=.2)
    #무한한 길이의 시퀀스를 줄 수는 없기 때문에
    #max_length가 늘리면 배치가 줄어야 함, max_length가 줄면 배치를 늘릴 수 있음
    #그러나 max_length가 너무 줄면 transformer의 학습능력이 줄어듦
    #하이퍼 파라미터이므로, 적절하게 적용해야 함
    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config

def get_loaders(fn, tokenizer, valid_ratio=.2):
    labels, texts = read_text(fn) #utils에 있음

    #unique label로 만들어 줌
    unique_labels = list( set(labels) ) 
    #맵 만들기
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    labels = list( map(label_to_index.get, labels) )

    #shuffl
    shuffled = list(
        zip(texts, labels) #묶어서 shuffle해야 엉키지 않음!!
        )
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio)) #train과 valid 나누는 길이

    train_loader = DataLoader(
        TextClassificationDataset(texts[:idx], label[:idx]),
        batch_size = config.batch_size,
        shuffle=True,
        collated_fn = TextClassificationCollator(tokenizer, config.max_length),
    )

    #shuffle 없음
    valid_loader = DataLoader(
        TextClassificationDataset(texts[:idx], label[:idx]),
        batch_size = config.batch_size,
        collated_fn = TextClassificationCollator(tokenizer, config.max_length),
    )

    return train_loader, valid_loader, index_to_label

#optimizer를 가져옴
def get_optimizer(model, config):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr = config.lr)
    else:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr = config.lr,
            eps = config.adam_epsilon
        )

    return optimizer

#main
def main(config):
    #pretrained BertTokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)

    #위에 있는 get_loaders에서 받아옴
    train_loader, valid_loader, index_to_label = get_loaders(
        config.train_fn,
        tokenizer,
        valid_ratio = config.valid_ratio
    )

    print(
        '|train| = ', len(train_loader) * config.batch_size,
        '|valid| = ', len(valid_loader) * config.batch_size,
    )

    #미니배치수 X 에폭
    n_total_iterations = len(train_loader) * config.n_epochs
    #warmup은 초기에 transformer가 안정적으로 학습되기 전까지 lr를 서서히 올리는 단계
    #default는 80
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

    print(
        '#total_iters = ', n_total_iterations,
        '#warmup_iters = ', n_warmup_steps,
    )

    #model불러오기
    model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
    #이름 넣으면 알아서 가져옴
    model = model_loader.from_pretrained(
        config.pretrained_model_name,
        #아웃풋 linear layer의 갯수를 정하는 것, ex) 긍정, 부정에 대한 classifier면 2
        num_labels = len(index_to_label) 
    )
    optimizer = get_optimizer(model, config)

    #crossentropyloss를 사용할 것임
    crit = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    #gpu로 옮기기
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    #trainer 선언
    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    #pickle로 감싸서 저장
    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer, #tokenizer 필수
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)