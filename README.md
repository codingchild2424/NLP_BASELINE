# [Introduce]
This repository contains natural language processing base line models for learning.


실행을 위한 명령어

# <Transformer(Not yet)>
python

# <BERT_native(Debugging)>
python finetune_plm_native.py --model_fn review.native.kcbert.pth --train_fn ../datasets/review.sorted.uniq.refined.shuf.train.tsv --gpu_id 0 --batch_size 80 --n_epochs 2

# <BERT_huggingface]>

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

위 명령어 설명
1. data를 불러오고
  cat ../datasets/review.sorted.uniq.refined.shuf.test.tsv
2. 2번째 column만 가져오고
  awk -F'\t' '{ print $2 }'
3. 모델을 가져오고
  python classify_plm.py --model_fn ./models/review.hft.kykim-bert.pth
4. GPU와 batch
  --gpu_id 0 --batch_size 32
=> 이렇게 하면 결과 값이 나옴, 이후 정답만 저장해서 원본 ground truth와 비교하는 작업
5. 먼저, 결과로 나온 값의 정답만 저장
  awk -F'\t' '{ print $1 }' > ./models/review.hft.kykim-bert.pth.result.txt
6. get_accuracy.py를 활용해서 모델의 결과로 나온 값과 원래 ground_truth 값을 비교
  python ./get_accuracy.py ./models/review.hft.kykim-bert.pth.result.txt ./models/ground_truth.result.txt
  *ground_truth값이 만약 없다면 아래 명령어를 통해 추출
    cat ../datasets/review.sorted.uniq.refined.shuf.test.tsv | awk -F'\t' '{ print $1 }' > ./models/ground_truth.result.txt

### GPU 상태 확인
watch -n .5 nvidia-smi

# <refernce>
https://github.com/kh-kim/simple-ntc