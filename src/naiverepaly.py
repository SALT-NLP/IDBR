import os
import argparse
from copy import deepcopy

DATA_DIR = '../data'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", nargs='+', type=int, default=[10, 10, 10, 10, 10])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--bert_learning_rate", type=float, default=3e-5)
parser.add_argument("--learning_rate", type=float, default=3e-5)

# 0822 1628 -- increase latent size, use tsne
parser.add_argument("--replay_freq", type=int, default=10)
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=2000,
                    help='Number of labeled data')
parser.add_argument("--store_ratio", type=float, default=0.01)
parser.add_argument('--tasks', nargs='+', type=str,
                    default=['ag', 'yelp', 'amazon', 'yahoo', 'dbpedia'], help='Task Sequence')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW

from model import BaseModel
from read_data import compute_class_offsets, prepare_dataloaders

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
n_gpu = torch.cuda.device_count()

dataset_classes = {
    'amazon'  : 5,
    'yelp'    : 5,
    'yahoo'   : 10,
    'ag'      : 4,
    'dbpedia' : 14,
}


class Memory(object):
    def __init__(self):
        self.examples = []
        self.labels = []
        self.tasks = []

    def append(self, example, label, task):
        self.examples.append(example)
        self.labels.append(label)
        self.tasks.append(task)

    def get_random_batch(self, batch_size, task_id=None):
        if task_id is None:
            permutations = np.random.permutation(len(self.labels))
            index = permutations[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
        else:
            index = [i for i in range(len(self.labels)) if self.tasks[i] == task_id]
            np.random.shuffle(index)
            index = index[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
        return torch.tensor(mini_examples), torch.tensor(mini_labels), torch.tensor(mini_tasks)

    def __len__(self):
        return len(self.labels)


def train_step(model, optimizer, cls_CR, x, y):
    model.train()
    logits = model(x)
    loss = cls_CR(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validation(model, t, validation_loaders):
    model.eval()
    acc_list = []
    with torch.no_grad():
        avg_acc = 0.0
        for i in range(t + 1):
            valid_loader = validation_loaders[i]
            total = 0
            correct = 0
            for x, mask, y in valid_loader:
                x, y = x.to(device), y.to(device)
                batch_size = x.size(0)
                logits = model(x)
                _, pred_cls = logits.max(1)
                correct += pred_cls.eq(y.view_as(pred_cls)).sum().item()
                total += batch_size
            print("acc on task {} : {}".format(i, correct * 100.0 / total))
            avg_acc += correct * 100.0 / total
            acc_list.append(correct * 100.0 / total)

    return avg_acc / (t + 1), acc_list


def random_select_samples_to_store(buffer, data_loader, task_id):
    x_list = []
    y_list = []
    for x, mask, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        x_list.append(x)
        y_list.append(y)
    x_list = torch.cat(x_list, dim=0).data.cpu().numpy()
    y_list = torch.cat(y_list, dim=0).data.cpu().numpy()
    permutations = np.random.permutation(len(x_list))
    index = permutations[:int(args.store_ratio * len(x_list))]
    for j in index:
        buffer.append(x_list[j], y_list[j], task_id)

    print("Buffer size:{}".format(len(buffer)))
    b_lbl = np.unique(buffer.labels)
    for i in b_lbl:
        print("Label {} in Buffer: {}".format(i, buffer.labels.count(i)))


def main():
    np.random.seed(0)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    task_num = len(args.tasks)
    task_classes = [dataset_classes[task] for task in args.tasks]
    total_classes, offsets = compute_class_offsets(args.tasks, task_classes)
    train_loaders, validation_loaders, test_loaders = \
        prepare_dataloaders(DATA_DIR, args.tasks, offsets, args.n_labeled,
                            2000, args.batch_size, 128, 128)

    # Reset random seed by the torch seed
    np.random.seed(torch.randint(1000, [1]).item())

    buffer = Memory()
    model = BaseModel(total_classes).to(args.device)
    cls_CR = torch.nn.CrossEntropyLoss()

    for task_id in range(task_num):
        data_loader = train_loaders[task_id]
        length = len(data_loader)

        optimizer = AdamW(
            [
                {"params": model.bert.parameters(), "lr": args.bert_learning_rate},
                {"params": model.classifier.parameters(), "lr": args.learning_rate},
            ]
        )

        best_acc = 0
        best_model = deepcopy(model.state_dict())

        acc_track = []

        for epoch in range(args.epochs[task_id]):
            iteration = 1
            for x, mask, y in tqdm(data_loader, total=length, ncols=100):
                if iteration % args.replay_freq == 0 and task_id > 0:
                    # replay once every args.replay_freq batches, starting from the 2nd task
                    total_x, total_y = x, y
                    for j in range(task_id):
                        old_x, old_y, old_t = buffer.get_random_batch(args.batch_size, j)
                        total_x = torch.cat([old_x, total_x], dim=0)
                        total_y = torch.cat([old_y, total_y], dim=0)
                    permutation = np.random.permutation(total_x.shape[0])
                    total_x = total_x[permutation, :]
                    total_y = total_y[permutation]
                    for j in range(task_id + 1):
                        x = total_x[j * args.batch_size: (j + 1) * args.batch_size, :]
                        y = total_y[j * args.batch_size: (j + 1) * args.batch_size]
                        x, y = x.to(device), y.to(device)
                        train_step(model, optimizer, cls_CR, x, y)
                else:
                    x, y = x.to(device), y.to(device)
                    train_step(model, optimizer, cls_CR, x, y)

                if iteration % 250 == 0:
                    print("----------------Validation-----------------")
                    avg_acc, acc_list = validation(model, task_id, validation_loaders)
                    acc_track.append(acc_list)

                    if avg_acc > best_acc:
                        print("------------------Best Model Till Now------------------------")
                        best_acc = avg_acc
                        best_model = deepcopy(model.state_dict())

                iteration += 1

            if task_id == 0:
                print("----------------Validation-----------------")
                avg_acc, _ = validation(model, task_id, validation_loaders)

                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_model = deepcopy(model.state_dict())
                    print("------------------Best Model Till Now------------------------")

        if len(acc_track) > 0:
            print("ACC Track: {}".format(acc_track))

        model.load_state_dict(deepcopy(best_model))
        print("------------------Best Result------------------")
        avg_acc, _ = validation(model, task_id, test_loaders)
        print("Best avg acc: {}".format(avg_acc))

        random_select_samples_to_store(buffer, data_loader, task_id)


if __name__ == '__main__':
    print(args)
    main()