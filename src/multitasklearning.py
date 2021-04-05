import os
import argparse
from copy import deepcopy

DATA_DIR = '../data'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=2,
                    help='Epoch number')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--bert_learning_rate", type=float, default=3e-5,
                    help='learning rate for pretrained Bert')
parser.add_argument("--learning_rate", type=float, default=3e-5,
                    help='learning rate for Class Classifier/General Space Encoder/Specific Space Encoder')
parser.add_argument("--task_learning_rate", type=float, default=5e-4,
                    help='learning rate for Task ID Classifier')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=2000,
                    help='Number of training data for each class')
parser.add_argument('--n-val', type=int, default=2000,
                    help='Number of validation data for each class')
parser.add_argument("--nspcoe", type=float, default=1.0,
                    help='Coefficient for Next Sentence Prediction Loss')
parser.add_argument("--tskcoe", type=float, default=1.0,
                    help='Coefficient for task ID Prediction Loss')
parser.add_argument("--disen", type=bool, default=False,
                    help='Apply Information Disentanglement or not')
parser.add_argument("--hidden_size", type=int, default=128,
                    help='size of General/Specific Space')
parser.add_argument('--tasks', nargs='+', type=str,
                    default=['ag', 'yelp', 'amazon', 'yahoo', 'dbpedia'], help='Task Sequence')
parser.add_argument('--select_best', type=bool, default=True,
                    help='whether picking the model with best val acc on each task')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW, get_constant_schedule_with_warmup

from model import Model, Predictor
from read_data import compute_class_offsets, prepare_dataloaders
from train import random_seq, random_string, change_string, get_permutation_batch

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
        self.masks = []
        self.labels = []
        self.tasks = []
        self.features = []

    def append(self, example, mask, label, task):
        self.examples.append(example)
        self.masks.append(mask)
        self.labels.append(label)
        self.tasks.append(task)

    def store_features(self, model):
        """

        Args:
            model: The model trained just after previous task

        Returns: None

        store previous features before trained on new class
        """
        self.features = []
        length = len(self.labels)
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(length), total=length, ncols=100):
                x = torch.tensor(self.examples[i]).view(1, -1).to(device)
                mask = torch.tensor(self.masks[i]).view(1, -1).to(device)
                g_fea, s_fea, _, _, _ = model(x, mask)
                fea = torch.cat([g_fea, s_fea], dim=1).view(-1).data.cpu().numpy()
                self.features.append(fea)

    def get_minibatch(self, batch_size):
        length = len(self.labels)
        permutations = np.random.permutation(length)
        for s in range(0, length, batch_size):
            if s + batch_size >= length:
                break
            index = permutations[s:s + batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
            yield torch.tensor(mini_examples), torch.tensor(mini_masks), torch.tensor(mini_labels), \
                  torch.tensor(mini_tasks), torch.tensor(mini_features)

    def __len__(self):
        return len(self.labels)

    def shuffle(self):
        length = len(self.labels)
        permutations = np.random.permutation(length)
        self.labels = [self.labels[i] for i in permutations]
        self.masks = [self.masks[i] for i in permutations]
        self.examples = [self.examples[i] for i in permutations]
        self.tasks = [self.tasks[i] for i in permutations]


def train_step(model, optimizer, nsp_CR, cls_CR, x, mask, y, t,
            predictor, optimizer_P, scheduler, scheduler_P):
    batch_size = x.size(0)

    model.train()
    predictor.train()

    model.zero_grad()
    predictor.zero_grad()

    x = random_seq(x)
    nsp_lbl = None
    if args.disen:
        p_x, p_mask, p_lbl = get_permutation_batch(x, mask)
        x = torch.cat([x, p_x], dim=0)
        mask = torch.cat([mask, p_mask], dim=0)
        r_lbl = torch.zeros_like(p_lbl)
        nsp_lbl = torch.cat([r_lbl, p_lbl], dim=0)

        y = torch.cat([y, y], dim=0)
        t = torch.cat([t, t], dim=0)

    total_g_fea, total_s_fea, cls_pred, task_pred, _ = model(x, mask)

    #Calculate classification loss
    _, pred_cls = cls_pred.max(1)
    correct_cls = pred_cls.eq(y.view_as(pred_cls)).sum().item()
    cls_loss = cls_CR(cls_pred, y)

    task_loss = torch.tensor(0.0).to(device)
    nsp_loss = torch.tensor(0.0).to(device)

    nsp_acc = 0.0
    correct_task = 0.0
    if args.disen:
        #Calculate Next Sentence Prediction loss
        nsp_output = predictor(total_g_fea)
        nsp_loss += args.nspcoe * nsp_CR(nsp_output, nsp_lbl)

        _, nsp_pred = nsp_output.max(1)
        nsp_correct = nsp_pred.eq(nsp_lbl.view_as(nsp_pred)).sum().item()
        nsp_acc = nsp_correct * 1.0 / (batch_size * 2.0)

        #Calculate task loss
        _, pred_task = task_pred.max(1)
        correct_task = pred_task.eq(t.view_as(pred_task)).sum().item()
        task_loss += args.tskcoe * cls_CR(task_pred, t)

    loss = cls_loss + task_loss + nsp_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    if args.disen:
        optimizer_P.step()
        scheduler_P.step()

    return nsp_acc, correct_cls, correct_task, nsp_loss.item(), task_loss.item(), cls_loss.item()


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
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                batch_size = x.size(0)
                g_fea, s_fea, cls_pred, _, _ = model(x, mask)
                _, pred_cls = cls_pred.max(1)
                correct += pred_cls.eq(y.view_as(pred_cls)).sum().item()
                total += batch_size
            print("acc on task {} : {}".format(i, correct * 100.0 / total))
            avg_acc += correct * 100.0 / total
            acc_list.append(correct * 100.0 / total)

    return avg_acc / (t + 1), acc_list


def main():
    # Fixed numpy random seed for dataset split
    np.random.seed(0)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    task_num = len(args.tasks)
    task_classes = [dataset_classes[task] for task in args.tasks]
    total_classes, offsets = compute_class_offsets(args.tasks, task_classes)
    train_loaders, validation_loaders, test_loaders = \
        prepare_dataloaders(DATA_DIR, args.tasks, offsets, args.n_labeled,
                            args.n_val, args.batch_size, 16, 16)

    # Reset random seed by the torch seed
    np.random.seed(torch.randint(1000, [1]).item())

    model = Model(
        n_tasks=task_num,
        n_class=total_classes,
        hidden_size=args.hidden_size).to(args.device)

    predictor = Predictor(2, hidden_size=args.hidden_size).to(args.device)

    nsp_CR = torch.nn.CrossEntropyLoss()
    cls_CR = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(
        [
            {"params": model.Bert.parameters(), "lr": args.bert_learning_rate, "weight_decay": 0.01},
            {"params": model.General_Encoder.parameters(), "lr": args.learning_rate, "weight_decay": 0.01},
            {"params": model.Specific_Encoder.parameters(), "lr": args.learning_rate, "weight_decay": 0.01},
            {"params": model.cls_classifier.parameters(), "lr": args.learning_rate, "weight_decay": 0.01},
            {"params": model.task_classifier.parameters(), "lr": args.task_learning_rate, "weight_decay": 0.01},
        ]
    )
    optimizer_P = AdamW(
        [
            {"params": predictor.parameters(), "lr": args.learning_rate, "weight_decay": 0.01},
        ]
    )
    # linear warmup
    scheduler = get_constant_schedule_with_warmup(optimizer, 1000)
    scheduler_P = get_constant_schedule_with_warmup(optimizer_P, 1000)

    best_acc = 0
    best_model = deepcopy(model.state_dict())
    best_predictor = deepcopy(predictor.state_dict())

    acc_track = []

    currentBuffer = Memory()
    print("INIT current buffer...")
    for task_id in range(len(args.tasks)):
        data_loader = train_loaders[task_id]
        with torch.no_grad():
            for x, mask, y in data_loader:
                for i, yi in enumerate(y):
                    currentBuffer.append(x[i].data.cpu().numpy(), mask[i].data.cpu().numpy(), y[i].item(), task_id)
    print("Start Storing Features...")
    currentBuffer.store_features(model)
    currentBuffer.shuffle()

    length = len(currentBuffer)

    for epoch in range(args.epochs):
        cls_losses = []
        nsp_losses = []
        tsk_accs = []
        cls_accs = []
        nsp_accs = []

        current_cls_losses = []
        current_nsp_losses = []
        current_tsk_accs = []
        current_cls_accs = []
        current_nsp_accs = []

        iteration = 1
        for x, mask, y, t, _ in tqdm(currentBuffer.get_minibatch(args.batch_size),
                                              total=length // args.batch_size, ncols=100):
            x, mask, y, t = x.to(device), mask.to(device), y.to(device), t.to(device)
            nsp_acc, correct_cls, correct_task, nsp_loss, task_loss, cls_loss, = \
                train_step(model, optimizer, nsp_CR, cls_CR, x, mask, y, t,
                           predictor, optimizer_P, scheduler, scheduler_P)

            current_cls_losses.append(cls_loss)
            current_nsp_losses.append(nsp_loss)

            current_tsk_accs.append(correct_task * 1.0 / x.size(0))
            current_cls_accs.append(correct_cls * 1.0 / x.size(0))
            current_nsp_accs.append(nsp_acc)

            if iteration % 250 == 0:
                print("----------------Validation-----------------")
                avg_acc, acc_list = validation(model, len(args.tasks) - 1, validation_loaders)
                acc_track.append(acc_list)

                if avg_acc > best_acc:
                    print("------------------Best Model Till Now------------------------")
                    best_acc = avg_acc
                    best_model = deepcopy(model.state_dict())
                    best_predictor = deepcopy(predictor.state_dict())

            iteration += 1

        if len(cls_losses) > 0:
            print("Mean CLS Loss: {}".format(np.mean(cls_losses)))
        if len(nsp_losses) > 0:
            print("Mean PRE Loss: {}".format(np.mean(nsp_losses)))
        if len(tsk_accs) > 0:
            print("Mean TSK Acc: {}".format(np.mean(tsk_accs)))
        if len(cls_accs) > 0:
            print("Mean LBL Acc: {}".format(np.mean(cls_accs)))
        if len(nsp_accs) > 0:
            print("Mean PRE Acc: {}".format(np.mean(nsp_accs)))

        if len(current_cls_losses) > 0:
            print("Mean Current CLS Loss: {}".format(np.mean(current_cls_losses)))
        if len(current_nsp_losses) > 0:
            print("Mean PRE Loss: {}".format(np.mean(current_nsp_losses)))
        if len(current_tsk_accs) > 0:
            print("Mean Current TSK Acc: {}".format(np.mean(current_tsk_accs)))
        if len(current_cls_accs) > 0:
            print("Mean Current LBL Acc: {}".format(np.mean(current_cls_accs)))
        if len(current_nsp_accs) > 0:
            print("Mean Current PRE Acc: {}".format(np.mean(current_nsp_accs)))

    if len(acc_track) > 0:
        print("ACC Track: {}".format(acc_track))

    if args.select_best:
        model.load_state_dict(deepcopy(best_model))
        predictor.load_state_dict(deepcopy(best_predictor))
    print("------------------Best Result------------------")
    avg_acc, _ = validation(model, len(args.tasks) - 1, test_loaders)
    print("Best avg acc: {}".format(avg_acc))


if __name__ == '__main__':
    print(args)
    main()
