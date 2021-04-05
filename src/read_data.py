import os
import pandas as pd
import numpy as np
import torch
from transformers import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_tokenized(texts, tokenizer, max_seq_len):
    result = []
    mask_res = []
    for text in texts:
        tokenized_res = tokenizer._tokenize(text)
        if len(tokenized_res) > max_seq_len - 3:
            tokenized_res = tokenized_res[:max_seq_len - 3]
        ids = tokenizer.convert_tokens_to_ids(tokenized_res)
        len1 = len(ids) // 2

        x = [101] + ids[:len1] + [102] + ids[len1:] + [102]
        mask = [1] * len(x)

        padding = [0] * (max_seq_len - len(x))
        x += padding
        mask += padding

        assert len(x) == max_seq_len
        assert len(mask) == max_seq_len

        result.append(x)
        mask_res.append(mask)
    return result, mask_res


def get_train_val_data(data_path, n_train_per_class=2000,
                       n_val_per_class=2000, max_seq_len=256,
                       model="bert-base-uncased", offset=0):
    tokenizer = BertTokenizer.from_pretrained(model)
    assert tokenizer.convert_tokens_to_ids('[PAD]') == 0, "token id error"
    assert tokenizer.convert_tokens_to_ids('[CLS]') == 101, "token id error"
    assert tokenizer.convert_tokens_to_ids('[MASK]') == 103, "token id error"

    data_path = os.path.join(data_path, 'train.csv')
    train_df = pd.read_csv(data_path, header=None)

    train_idxs, val_idxs = train_val_split(train_df, n_train_per_class,
                                           n_val_per_class)
    train_labels, train_text = get_data_by_idx(train_df, train_idxs)
    val_labels, val_text = get_data_by_idx(train_df, val_idxs)

    train_labels = [label + offset for label in train_labels]
    val_labels = [label + offset for label in val_labels]

    train_text = get_tokenized(train_text, tokenizer, max_seq_len)
    val_text = get_tokenized(val_text, tokenizer, max_seq_len)

    train_dataset = myDataset(train_text, train_labels)
    val_dataset = myDataset(val_text, val_labels)

    print("#Train: {}, Val: {}".format(len(train_idxs), len(val_idxs)))
    return train_dataset, val_dataset


def get_test_data(data_path, max_seq_len=256,
                  model='bert-base-uncased', offset=0):
    tokenizer = BertTokenizer.from_pretrained(model)
    data_path = os.path.join(data_path, 'test.csv')
    test_df = pd.read_csv(data_path, header=None)
    test_idxs = list(range(test_df.shape[0]))
    np.random.shuffle(test_idxs)

    test_labels, test_text = get_data_by_idx(test_df, test_idxs)

    test_labels = [label + offset for label in test_labels]
    test_text = get_tokenized(test_text, tokenizer, max_seq_len)

    print("#Test: {}".format(len(test_labels)))
    test_dataset = myDataset(test_text, test_labels)
    return test_dataset


def train_val_split(train_df, n_train_per_class, n_val_per_class, seed=0):
    np.random.seed(seed)
    train_idxs = []
    val_idxs = []

    min_class = min(train_df[0])
    max_class = max(train_df[0])
    for cls in range(min_class, max_class + 1):
        idxs = np.array(train_df[train_df[0] == cls].index)
        np.random.shuffle(idxs)
        train_pool = idxs[:-n_val_per_class]
        if n_train_per_class < 0:
            train_idxs.extend(train_pool)
        else:
            train_idxs.extend(train_pool[:n_train_per_class])
        val_idxs.extend(idxs[-n_val_per_class:])

    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


def get_data_by_idx(df, idxs):
    text = []
    labels = []
    for item_id in idxs:
        labels.append(df.loc[item_id, 0] - 1)
        text.append(df.loc[item_id, 2])
    return labels, text


def prepare_dataloaders(data_dir, tasks, offsets, train_class_size,
                        val_class_size, train_batch_size, val_batch_size,
                        test_batch_size):
    task_num = len(tasks)
    train_loaders = []
    validation_loaders = []
    test_loaders = []

    for i in range(task_num):
        data_path = os.path.join(data_dir, tasks[i])
        train_dataset, val_dataset = \
            get_train_val_data(data_path, train_class_size,
                               val_class_size, offset=offsets[i])
        test_dataset = get_test_data(data_path, offset=offsets[i])
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  shuffle=True, drop_last=True)
        validation_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                                       shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                                 shuffle=True, drop_last=True)
        train_loaders.append(train_loader)
        validation_loaders.append(validation_loader)
        test_loaders.append(test_loader)

    return train_loaders, validation_loaders, test_loaders


class myDataset(Dataset):

    def __init__(self, data, labels):
        super(myDataset, self).__init__()
        self.text = data[0]
        self.mask = data[1]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.text[idx]), \
               torch.tensor(self.mask[idx]), self.labels[idx]


def compute_class_offsets(tasks, task_classes):
    '''
    :param tasks: a list of the names of tasks, e.g. ["amazon", "yahoo"]
    :param task_classes:  the corresponding numbers of classes, e.g. [5, 10]
    :return: the class # offsets, e.g. [0, 5]
    Here we merge the labels of yelp and amazon, i.e. the class # offsets
    for ["amazon", "yahoo", "yelp"] will be [0, 5, 0]
    '''
    task_num = len(tasks)
    offsets = [0] * task_num
    prev = -1
    total_classes = 0
    for i in range(task_num):
        if tasks[i] in ["amazon", "yelp"]:
            if prev == -1:
                prev = i
                offsets[i] = total_classes
                total_classes += task_classes[i]
            else:
                offsets[i] = offsets[prev]
        else:
            offsets[i] = total_classes
            total_classes += task_classes[i]
    return total_classes, offsets