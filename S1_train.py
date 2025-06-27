import os
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import wcat as at
from transformers import GPT2Tokenizer
import h5py
import csv
import time

Main_ROIs=['V1', 'MST', 'V6', 'V2', 'V3', 'V4', 'V8', 'V3A', 'V7', 'IPS1', 'FFC', 'V3B', 'LO1', 'LO2', 'PIT',
           'MT', 'PCV', 'STV', 'PH', 'DVT', 'V6A', 'VMV1', 'VMV3', 'V4t', 'FST', 'V3CD', 'LO3', 'VMV2', 'VVC']
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/share/home/huafuchen01/LJP/decodeGPT/add_pad", type=str, help='')
    parser.add_argument('--save_model_path', default="/share/home/huafuchen01/LJP/decodeGPT/wudiwei_ablation/save_models/using_GRU", type=str, help='')
    parser.add_argument('--log_path', default="/share/home/huafuchen01/LJP/decodeGPT/wudiwei_ablation/save_models/using_GRU", type=str,
                        help='')
    parser.add_argument('--train_raw_path', default='/share/home/huafuchen01/LJP/decodeGPT/txt/trainData_S1.txt', type=str, help='')
    parser.add_argument('--batch_size', default=64, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=5000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1.5e-3, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=1, type=int, required=False, help='print log steps')
    parser.add_argument('--save_epoch', default=1, type=int, required=False, help='save epochs')

    return parser.parse_args()

def load_model(model_path):
    model = at.decodeGPTLMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    return model, tokenizer

def train_data_loader(args, train_data_path, tokenizer, shuffle):
    text_list = []
    subj_list = []
    stms_list = []
    lastsubj = 'subj00'#初始化上一个被试编号
    with open(train_data_path, 'rb') as f:
        data = f.read().decode("utf-8")
        train_data = data.split("\n")
        train_data.pop()#删除最后空行
        print("数据总行数:{}".format(len(train_data)))
        # read task and label
        load_num = 0
        for txt in tqdm(train_data):
            text_split = txt.split("#", 2)
            stm, subjid, text = text_split
            subj='subj' + subjid
            subj_stm_path = '/share/home/huafuchen01/huangwei/WORKS/Task00_dataset/Natural-Scenes-Dataset/NSD-Code/step00_excel/trailID-' + subj + '.csv'


            #如果被试编号变化，重新定位数据集位置
            if( subj != lastsubj ):
                #打开刺激列表
                stm_data = []
                with open(subj_stm_path) as csvfile:
                    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                    for row in csv_reader:  # 将csv 文件中的数据保存到data中
                        stm_data.append(row[11])  # 选择某一列加入到data数组中
                stm_data.pop(0) #去除列标签
            lastsubj = subj#更新上一个被试编号


            if ( stm_data[int(stm)] == 'False'):
                load_num = load_num + 1
                text_ids = tokenizer.encode('<|endoftext|>' + text + '<|endoftext|>')
                text_list.append(text_ids)
                # read rois data
                subj_list.append(subj)
                stms_list.append(int(stm))

    print(str(load_num)+' train samples loaded')
    dataset = MyDataset(text_list, subj_list, stms_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader

class MyDataset(Dataset):
    def __init__(self, text_list, subj_list, stms_list):
        self.text_list = text_list
        self.subj_list = subj_list
        self.stms_list = stms_list


    def __getitem__(self, index):
        text_ids = self.text_list[index]
        subj_id = self.subj_list[index]
        fmri_stm = self.stms_list[index]

        return text_ids, subj_id, fmri_stm

    def __len__(self):
        return len(self.text_list)


def collate_fn(batch):
    text_len_list = []
    for btc_idx in range(len(batch)):
        text_len = len(batch[btc_idx][0])
        text_len_list.append(text_len)
    max_text_len = max(text_len_list)
    data = []
    for btc_idx in range(len(batch)):
        text_len = len(batch[btc_idx][0])
        data.append(list(batch[btc_idx]))
        # use 'padding' to pad
        data[btc_idx][0].extend([50257] * (max_text_len - text_len))
        data[btc_idx][0] = torch.tensor(data[btc_idx][0], dtype=torch.long)
    return data

def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=50257, reduction='sum')
    pre = shift_logits.view(-1, shift_logits.size(-1))
    gt = shift_labels.view(-1)
    loss = loss_fct(pre, gt)

    _, preds = shift_logits.max(dim=-1)

    not_ignore = shift_labels.ne(50257)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    return loss, accuracy



def train(args, model, tokenizer, dataloader):
    num_training_steps = args.epochs * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    batch_steps = 0
    cur_epochs = 0

    loss_list=[]
    accuracy_list=[]
    begin = time.perf_counter()

    # 读取4个被试roi数据到内存

    subjs = ['subj01', 'subj02', 'subj05', 'subj07']
    str_to_int = {"subj01": 1, "subj02": 2, "subj05": 5, "subj07": 7}
    # 读取fmri_emb
    with h5py.File('/share/home/huafuchen01/LJP/decodeGPT/fmri_emb/fmri_emb.h5', 'r') as hf:
        # 加载 main_data
        main_data = {subj: torch.FloatTensor(hf[f'main_data/{subj}'][:]) for subj in subjs}

        # 加载 other_data
        other_data = {subj: torch.FloatTensor(hf[f'other_data/{subj}'][:]) for subj in subjs}

    for epoch in range(args.epochs):
        cur_epochs += 1
        for batch in dataloader:
            batch_steps += 1
            text_ids = torch.LongTensor(len(batch), len(batch[0][0]))
            main_fmri_embeds = torch.FloatTensor(len(batch), 26, 768)
            other_fmri_embeds = torch.FloatTensor(len(batch), 113, 768)
            subj_ids = torch.FloatTensor(len(batch), 1)
            for btc_idx in range(len(batch)):
                text_ids[btc_idx, :] = batch[btc_idx][0]
                subj_id = batch[btc_idx][1]
                subj_ids[btc_idx, :] = str_to_int.get(batch[btc_idx][1], -1)
                stm = batch[btc_idx][2]
                # 根据subj与stm索引读取fmri信号
                main_fmri_embeds[btc_idx, :, :]=main_data[subj_id][stm, :, :]
                other_fmri_embeds[btc_idx, :, :] = other_data[subj_id][stm, :, :]
            inputs = {"input_ids": text_ids.to(device),
                      "subj_ids": subj_ids.to(device),
                      "main_fmri_embeds": main_fmri_embeds.to(device),
                      "other_fmri_embeds": other_fmri_embeds.to(device)}


            outputs = model(**inputs, labels=text_ids.to(device))

            # loss = outputs.loss
            loss, acc= calculate_loss_and_accuracy(outputs,text_ids.to(device), device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


            if batch_steps % args.log_step == 0:
                end = time.perf_counter()
                cost_time = end - begin
                begin = time.perf_counter()
                loss_list.append(loss.cpu().detach().numpy())
                accuracy_list.append(acc.cpu().detach().numpy())
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}, timecost {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc, cost_time))

        if cur_epochs % args.save_epoch == 0:
            if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.save_model_path + '/epoch' + str(cur_epochs), safe_serialization = False)
            tokenizer.save_pretrained(args.save_model_path + '/epoch' + str(cur_epochs))

            if not os.path.exists(args.log_path):
                os.makedirs(args.log_path)
            log_file = args.log_path + '/epoch' + str(cur_epochs) + '.h5'
            f = h5py.File(log_file, "w")
            f.create_dataset("loss_list", data=loss_list)
            f.create_dataset("acc_list", data=accuracy_list)
            f.close()
            loss_list=[]
            accuracy_list=[]



def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    args = setup_args()
    model, tokenizer = load_model(args.model_path)
    train_dataloader = train_data_loader(args, args.train_raw_path, tokenizer=tokenizer, shuffle=True)
    train(args, model, tokenizer, train_dataloader)
    # model, tokenizer = load_model(args.save_model_path)
    # eval_dataloader = train_data_loader(args, args.train_raw_path, Main_ROIs, tokenizer=tokenizer, shuffle=True)
    # evaluate(args, model, eval_dataloader)



