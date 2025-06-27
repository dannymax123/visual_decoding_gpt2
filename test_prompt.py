import os
import time
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import test_wcat as at
from transformers import GPT2Tokenizer
import h5py
import csv
import time
import glob

Main_ROIs = ['V1', 'MST', 'V6', 'V2', 'V3', 'V4', 'V8', 'V3A', 'V7', 'IPS1', 'FFC', 'V3B', 'LO1', 'LO2', 'PIT',
           'MT', 'PCV', 'STV', 'PH', 'DVT', 'V6A', 'VMV1', 'VMV3', 'V4t', 'FST', 'V3CD', 'LO3', 'VMV2', 'VVC']

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_path', default=r"/share/home/huafuchen01/LJP/decodeGPT/wudiwei_ablation/save_models/using_GRU", type=str, help='')
    parser.add_argument('--eval_raw_path', default=r'/share/home/huafuchen01/LJP/decodeGPT/txt/testData_S1.txt', type=str, help='')
    parser.add_argument('--batch_size', default=64, type=int, required=False, help='batch size')
    parser.add_argument('--save_pred_results_path', default=r'/share/home/huafuchen01/LJP/decodeGPT/wudiwei_ablation/result/using_GRU', type=str, help='')
    return parser.parse_args()


def load_model(model_path):
    model = at.decodeGPTLMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    return model, tokenizer




class MytestDataset(Dataset):
    def __init__(self, task_list, label_list, subj_list, stm_list):
        self.task_list = task_list
        self.label_list = label_list
        self.subj_list = subj_list
        self.stm_list = stm_list




    def __getitem__(self, index):
        task_ids = self.task_list[index]
        label_ids = self.label_list[index]
        subj = self.subj_list[index]
        stm = self.stm_list[index]

        return task_ids, label_ids, subj, stm

    def __len__(self):
        return len(self.task_list)

def collate_fn(batch):
    return batch

def test_data_loader(args, test_data_path, shuffle):
    task_list = []
    label_list = []
    subj_list = []
    stm_list = []
    lastsubj = 'subj00'  # 初始化上一个被试编号
    with open(test_data_path, 'rb') as f:
        data = f.read().decode("utf-8")
        train_data = data.split("\n")
        train_data.pop()  # 删除最后空行
        print("数据总行数:{}".format(len(train_data)))
        # read task and label
        load_num = 0
        for txt in tqdm(train_data):

            txt_split = txt.split("#", 2)
            # if len(text_split) != 2:
            #     continue
            stm, subjid, text = txt_split
            text_split = text.split(":", 1)
            task, label = text_split
            task = task + ':'
            subj = 'subj' + subjid
            subj_stm_path = '/share/home/huafuchen01/huangwei/WORKS/Task00_dataset/Natural-Scenes-Dataset/NSD-Code/step00_excel/trailID-' + subj + '.csv'


            # 如果被试编号变化，重新定位数据集位置
            if (subj != lastsubj):
                stm_data = []
                with open(subj_stm_path) as csvfile:
                    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                    for row in csv_reader:  # 将csv 文件中的数据保存到data中
                        stm_data.append(row[11])  # 选择某一列加入到data数组中
                stm_data.pop(0)  # 去除列标签


            lastsubj = subj  # 更新上一个被试编号

            if (stm_data[int(stm)] == 'True'):
                load_num = load_num + 1
                task_list.append('<|endoftext|>' + task)
                label_list.append(label)
                subj_list.append(subj)
                stm_list.append(int(stm))

    print(str(load_num) + ' test samples loaded')
    dataset = MytestDataset(task_list, label_list, subj_list, stm_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn= collate_fn)

    return dataloader

def predict(task, subj_id, main_fmri_embed, other_fmri_embed, model, tokenizer, device):
    max_length = 100
    caption = ''
    str_to_int = {"subj01": 1, "subj02": 2, "subj05": 5, "subj07": 7}
    for i in range(max_length):
        task_ids = []
        task_ids.extend(tokenizer.encode(task + caption))
        task_tensor = torch.tensor([task_ids], dtype=torch.long)
        subj_id_tensor = torch.tensor([str_to_int.get(subj_id, -1)], dtype=torch.float)
        inputs = {"input_ids": task_tensor.to(device), "subj_ids": subj_id_tensor, "main_fmri_embeds": main_fmri_embed.to(device),
                  "other_fmri_embeds": other_fmri_embed.to(device)}
        outputs = model(**inputs)
        logits = outputs.logits
        last_token_id = int(np.argmax(logits[0][-1].detach().to('cpu').numpy()))
        if last_token_id == 50256:# <endoftext>
            break
        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        last_token = last_token.replace('Ġ', ' ')
        task_ids.append(last_token_id)
        caption += last_token
    return caption


def predict_and_save(args):
    # 读取4个被试roi数据到内存

    subjs = ['subj01', 'subj02', 'subj05', 'subj07']

    with h5py.File('/share/home/huafuchen01/LJP/decodeGPT/fmri_emb/fmri_emb.h5', 'r') as hf:
        # 加载 main_data
        main_data = {subj: torch.FloatTensor(hf[f'main_data/{subj}'][:]) for subj in subjs}

        # 加载 other_data
        other_data = {subj: torch.FloatTensor(hf[f'other_data/{subj}'][:]) for subj in subjs}

    for i in range(5, 6):
        model_path = args.save_model_path + r'/epoch' + str(i)
        save_path = args.save_pred_results_path + r'/epoch' + str(i) + r'.txt'
        model, tokenizer = load_model(model_path)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        predict_dataloader = test_data_loader(args, args.eval_raw_path, shuffle=False)
        num_testing_steps = len(predict_dataloader)
        batch_steps = 0

        with open(save_path, "w") as f:
            for batch in predict_dataloader:
                time1 = time.time()
                batch_steps += 1
                for btc_idx in range(len(batch)):
                    task = batch[btc_idx][0]
                    label = batch[btc_idx][1]
                    subj_id = batch[btc_idx][2]
                    stm_id = batch[btc_idx][3]
                    # 根据subj与stm索引读取fmri信号
                    main_fmri_embed = main_data[subj_id][stm_id, :, :].unsqueeze(0)
                    other_fmri_embed = other_data[subj_id][stm_id, :, :].unsqueeze(0)
                    caption = predict(task, subj_id, main_fmri_embed, other_fmri_embed, model, tokenizer, device)
                    f.write('*INFO:' + subj_id + ' ' + str(stm_id) + '\n')
                    f.write('LABEL:' + label + '\n')
                    f.write('PRED:' + caption + "\n")
                    print('*INFO:' + subj_id + ' ' + str(stm_id) + '\n')
                    print('LABEL:' + label + '\n')
                    print('PRED:' + caption + "\n")

            print("epoch{} batch {}/{} time cost: {}\n".format(i, batch_steps, num_testing_steps, time.time() - time1))

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    predict_and_save(args)




