import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式环境需要这个
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from config import NerConfig
from model import BertBiLSTMNer,BertBiGRUNer,BiGRUNer,BiLSTMNer,BertCrfNer
from data_loader import NerDataset

from tqdm import tqdm
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
import re
import swanlab
import random

# 初始化一个新的swanlab run类来跟踪这个脚本
swanlab.init(
  # 设置将记录此次运行的项目信息
  project="my-awesome-project",
  workspace="yangning",
  # 跟踪超参数和运行元数据
  config={
        "max_seq_len":512,
        "epochs":1,
        "train_batch_size":12,
        "dev_batch_size":12,
        "bert_learning_rate":3e-5,
        "crf_learning_rate":3e-3,
        "adam_epsilon":1e-8,
        "weight_decay":0.01,
        "warmup_proportion":0.01,
  }
)

class Trainer:
    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 dev_loader=None,
                 test_loader=None,
                 optimizer=None,
                 schedule=None,
                 epochs=25,
                 device="cpu",
                 id2label=None):
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.schedule = schedule
        self.id2label = id2label
        self.total_step = len(self.train_loader) * self.epochs
        self.best_f1 = 0.0
        
        # 新增指标记录
        self.train_losses = []
        self.val_metrics = {
            'accuracy': [],
            'recall': [],
            'f1': []
        }

    def validate(self):
        self.model.eval()
        preds = []
        trues = []
        for step, batch_data in enumerate(tqdm(self.test_loader)):
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]
            output = self.model(input_ids, attention_mask, labels)
            logits = output.logits
            attention_mask = attention_mask.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = sum(attention_mask[i])
                logit = logits[i][1:length]
                logit = [self.id2label[i] for i in logit]
                label = labels[i][1:length]
                label = [self.id2label[i] for i in label]
                preds.append(logit)
                trues.append(label)

        report = classification_report(trues, preds,digits=4)
        lines = report.split('\n')
        for line in lines[2:-5]:  # 跳过标题行和平均行
            values = re.findall(r'\d+\.\d+', line)
            if values:
                print(values)
                current_f1 = float(values[2])
                current_accuracy = float(values[0])
                current_recall = float(values[1])
        # 记录指标
        self.val_metrics['accuracy'].append(current_accuracy)
        self.val_metrics['recall'].append(current_recall)
        self.val_metrics['f1'].append(current_f1)

        print(f"\n验证结果:")
        print(f"准确率: {current_accuracy:.4f} | 召回率: {current_recall:.4f} | F1: {current_f1:.4f}")
        print(f"历史最佳F1: {self.best_f1:.4f}")
        
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            print(f"发现更好的模型！保存最佳模型，F1: {current_f1:.4f}")
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model_bert_bilstm_crf.bin"))
        print("-" * 50)
        return current_accuracy,current_recall,current_f1

    def train(self):
        global_step = 1
        for epoch in range(1, self.epochs + 1):
            epoch_losses = []
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]
                output = self.model(input_ids, attention_mask, labels)
                loss = output.loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
                
                self.train_losses.append(loss.item())
                epoch_losses.append(loss.item())
                
                print(f"【train】{epoch}/{self.epochs} {global_step}/{self.total_step} loss:{loss.item():.4f}")

                current_accuracy,current_recall,current_f1 = self.validate()
                swanlab.log({"acc": current_accuracy,"recall": current_recall,"f1": current_f1})
                swanlab.log({"loss": loss.item()}) 
            # 每个epoch结束时保存训练损失平均值
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch} 平均训练损失: {avg_epoch_loss:.4f}")

    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        # 训练损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.title("Training Loss Curve")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid(True)
        
        # 验证指标曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.val_metrics['accuracy'], label='Accuracy')
        plt.plot(self.val_metrics['recall'], label='Recall')
        plt.plot(self.val_metrics['f1'], label='F1 Score')
        plt.title("Validation Metrics")
        plt.xlabel("Validation Steps")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join('./photo/training_metrics.png'))
        plt.close()

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "best_model_bert_bilstm_crf.bin")))
        self.model.eval()
        preds = []
        trues = []
        all_labels = list(self.id2label.values())
        
        for step, batch_data in enumerate(tqdm(self.test_loader)):
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]
            output = self.model(input_ids, attention_mask, labels)
            logits = output.logits
            attention_mask = attention_mask.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = sum(attention_mask[i])
                logit = logits[i][1:length]
                logit = [self.id2label[i] for i in logit]
                label = labels[i][1:length]
                label = [self.id2label[i] for i in label]
                preds.append(logit)
                trues.append(label)

        # 生成分类报告
        report = classification_report(trues, preds, digits=4)
        
        # 生成混淆矩阵
        flat_trues = [label for seq in trues for label in seq]
        flat_preds = [label for seq in preds for label in seq]
        
        cm = confusion_matrix(flat_trues, flat_preds, labels=all_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_labels, 
                    yticklabels=all_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join('./photo/confusion_matrix.png'))
        plt.close()

        return report



def build_optimizer_and_scheduler(args, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.bert_learning_rate},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.crf_learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def main(data_name):
    args = NerConfig(data_name)

    with open(os.path.join(args.output_dir, "ner_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.data_path, "train.txt"), "r", encoding="utf-8") as fp:
        train_data = fp.read().split("\n")
    train_data = [json.loads(d) for d in train_data]

    with open(os.path.join(args.data_path, "dev.txt"), "r", encoding="utf-8") as fp:
        dev_data = fp.read().split("\n")
    dev_data = [json.loads(d) for d in dev_data]

    train_dataset = NerDataset(train_data, args, tokenizer)
    dev_dataset = NerDataset(dev_data, args, tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=2)



    
    model = BertBiLSTMNer(args)

    # model = BertBiGRUNer(args)

    # model = BiGRUNer(args)

    # model = BiLSTMNer(args)
    
    # model = BertCrfNer(args)

    # for name,_ in model.named_parameters():
    #   print(name)

    model.to(device)
    t_toal = len(train_loader) * args.epochs
    optimizer, schedule = build_optimizer_and_scheduler(args, model, t_toal)

    train = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        optimizer=optimizer,
        schedule=schedule,
        epochs=args.epochs,
        device=device,
        id2label=args.id2label
    )

    train.train()

    report = train.test()
    
    print(report)
    
    train.plot_metrics()

if __name__ == "__main__":
    data_name = "dgre"
    main(data_name)
