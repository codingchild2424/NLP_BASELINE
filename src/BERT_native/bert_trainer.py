import torch
import torch.nn.utils as torch_utils

from ignite.engine import Events

from utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

from trainer import Trainer, MyEngine

#trainer.MyEngine 상속
class EngineForBert(MyEngine):

    def __init__(self, func, model, crit, optimizer, scheduler, config):
        self.scheduler = scheduler

        super().__init__(func, model, crit, optimizer, config)

    #아래의 BertTrainer에서 불러서 사용할 예정
    @staticmethod
    def train(engine, mini_batch):
        #train 모드
        engine.model.train()
        #optimizer 이니셜라이즈
        engine.optimizer.zero_grad()

        #미니배치에서 input_ids에 들어있는 텐서와 labels을 불러오기
        x, y = mini_batch['input_ids'], mini_batch['labels']
        #gpu로 옮기기
        x, y = x.to(engine.device), y.to(engine.device)
        #마스크 받아보기
        mask = mini_batch['attention_mask']
        #마스크 gpu 옮기기
        mask = mask.to(engine.device)

        #이미 데이터가 concat이 되어있겠지만, 혹시나를 위해
        #x는 (n, l, 1)이나 (n, l)로 되어있음, 따라서 1번 차원을 기준으로 슬라이싱
        x = x[:, :engine.config.max_length]

        #feed forward
        y_hat = engine.model(x, attention_mask = mask).logits 
        #.logits은 transformer에서 softmax 넣기 직전의 linear를 통과한 값임, 즉 linear를 통과한 값이 나옴
        #|y_hat| = (n, c)
        
        #crit은 crossentropy이기에 y_hat을 logit으로 처리해도 됨
        #왜냐면, 어차피 crossentropy값에서 y_hat에 softmax를 씌움
        loss = engine.crit(y_hat, y)
        #backpropagation
        loss.backward()

        #필요한 수치, accuracy
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim = -1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        #parameter의 L2 norm
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        #grad의 L2 norm
        g_norm = float(get_grad_norm(engine.model.parameters()))

        #optimzier step!
        engine.optimizer.step()
        #
        engine.scheduler.step()

        return {
            'loss': float(accuracy),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    #아래의 BertTrainer에서 불러서 사용할 예정
    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            #train과 같음
            x, y = mini_batch['input_ids'], mini_batch['labels']
            x, y = x.to(engine.device)  , y.to(engine.device)
            mask = mini_batch['attention_mask']
            mask = mask.to(engine.device)

            x = x[:, :engine.config.max_length]

            y_hat = engine.model(x, attention_mask = mask).logits

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) ==y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }

#trainer.Trainer 상속
#위에서 EngineForBert는 trainer.MyEngine을 상속받음, 이걸 여기에서 활용
class BertTrainer(Trainer):

    def __init__(self, config):
        self.config = config #학습을 위한 hyperparameter들을 가져옴

    def train(
        self,
        model, #bert
        crit, #criterion
        optimizer, #adam이나 radam
        scheduler,
        train_loader, valid_loader,
    ):
        #train_engine 등록
        train_engine = EngineForBert(
            #매 iteration마다 위에 있는 EngineForBert에서 train 매쏘드를 불러옴
            EngineForBert.train,
            model, crit, optimizer, scheduler, self.config
        )
        #validation_engine 등록
        validation_engine = EngineForBert(
            EngineForBert.validate,
            model, crit, optimizer, scheduler, self.config
        )

        #attach는 trainer.Trainer에서 상속받은 것임
        #train engine과 valid engine이 학습 중에 일어나는 일(progress bar와 결과값 출력)을 수행
        EngineForBert.attach(
            train_engine,
            validation_engine,
            verbose = self.config.verbose
        )

        #train의 매 epoch이 끝날때마다 valid engine이 실행되어야 하므로, 그것을 해주는 함수를 만듦
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs = 1)

        #위의 함수를 등록함
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #epoch이 끝날때마다
            run_validation, #run_validation 함수를 실행하고
            #두개의 아규먼트를 넣어줌
            validation_engine, #train_engine안에 validation_engine이 포함된 것을 확인할 수 있음
            valid_loader, 
        )

        #train_engine 안에 있는 상태인 것임
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            EngineForBert.check_best, #trainer.MyEngine에 있음
        )

        #여기서 훈련시킴
        #즉, train_engine만 실행시키면 안에 valid engine을 포함하고 있으므로 동시에 실행됨
        train_engine.run(
            train_loader,
            max_epochs = self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model