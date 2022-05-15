# Introduce
This repository contains natural language processing base line models for learning.

# Transformer(Not yet)

# BERT_native(Debugging)
python finetune_plm_native.py --model_fn review.native.kcbert.pth --train_fn ../datasets/review.sorted.uniq.refined.shuf.train.tsv --gpu_id 0 --batch_size 80 --n_epochs 2

# BERT_huggingface(Success)

## training
python finetune_plm_hftrainer.py --model_fn review.hft.kykim-bert.pth --train_fn ../datasets/review.sorted.uniq.refined.shuf.train.tsv --pretrained_model_name 'kykim/bert-kor-base' --n_epochs 1

## inference
### 데이터셋 확인
head ../datasets/review.sorted.uniq.refined.shuf.test.tsv

### 데이터셋에서 label이 아닌 데이터 부분만 가져오기
cat ../datasets/review.sorted.uniq.refined.shuf.test.tsv | awk -F'\t' '{ print $2 }' | head

### 추론 실행(nomal)
cat ../datasets/review.sorted.uniq.refined.shuf.test.tsv | awk -F'\t' '{ print $2 }' | head -n 30 | python classify_plm.py --model_fn ./models/review.hft.kykim-bert.pth --gpu_id 0
### 추론 실행(shuffling)
cat ../datasets/review.sorted.uniq.refined.shuf.test.tsv | awk -F'\t' '{ print $2 }' | shuf | head -n 30 | python classify_plm.py --model_fn ./models/review.hft.kykim-bert.pth --gpu_id 0
### 추론 실행 후 결과 확인 및 기록(Debugging)
cat ../datasets/review.sorted.uniq.refined.shuf.test.tsv | awk -F'\t' '{ print $2 }' | python classify_plm.py --model_fn ./models/review.hft.kykim-bert.pth --gpu_id 0 --batch_size 32 | awk -F'\t' '{ print $1 }' > ./models/review.hft.kykim-bert.pth.result.txt ; python ./get_accuracy.py ./models/review.hft.kykim-bert.pth.result.txt ./models/ground_truth.result.txt

### GPU 상태 확인
watch -n .5 nvidia-smi

# refernce
https://github.com/kh-kim/simple-ntc