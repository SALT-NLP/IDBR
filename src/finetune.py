import os
import argparse
from copy import deepcopy

DATA_DIR = '../data'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", nargs='+', type=int,
                    default=[10, 10, 10, 10, 10])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--bert_learning_rate", type=float, default=3e-5)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=2000,
                    help='Number of labeled data')
parser.add_argument('--tasks', nargs='+', type=str,
                    default=['ag', 'yelp', 'amazon', 'yahoo', 'dbpedia'],
                    help='Task Sequence')

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


def train_step(model, optimizer, cls_CR, x, y):
    model.train()
    logits = model(x)
    loss = cls_CR(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validation(model, t, validation_loaders):
    '''
    Compute the validation accuracy on the first (t + 1) tasks,
    return the average accuracy over (t + 1) tasks and detailed accuracy
    on each task.
    '''
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

        if len(acc_track) > 0:
            print("ACC Track: {}".format(acc_track))

        model.load_state_dict(deepcopy(best_model))
        print("------------------Best Result------------------")
        avg_acc, _ = validation(model, task_id, test_loaders)
        print("Best avg acc: {}".format(avg_acc))


if __name__ == '__main__':
    print(args)
    main()