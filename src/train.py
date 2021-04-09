import os
import argparse
from copy import deepcopy

DATA_DIR = '../data'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", nargs='+', type=int, default=[10, 10, 10, 10, 10],
                    help='Epoch number for each task')
parser.add_argument("--batch_size", type=int, default=8,
                    help='training batch size')
parser.add_argument("--bert_learning_rate", type=float, default=3e-5,
                    help='learning rate for pretrained Bert')
parser.add_argument("--learning_rate", type=float, default=3e-5,
                    help='learning rate for Class Classifier/General Space Encoder/Specific Space Encoder')
parser.add_argument("--task_learning_rate", type=float, default=5e-4,
                    help='learning rate for Task ID Classifier')
parser.add_argument("--replay_freq", type=int, default=10,
                    help='frequency of replaying, i.e. replay one batch from memory'
                         ' every replay_freq batches')
parser.add_argument('--kmeans', type=bool, default=False,
                    help='whether applying Kmeans when choosing examples to store')
parser.add_argument("--dump", type=bool, default=False,
                    help='dump the model or not')
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
parser.add_argument("--model_path", type=str, default="./dump",
                    help='where to dump the model')

parser.add_argument("--reg", type=bool, default=False,
                    help='Apply Regularization or Not')
parser.add_argument("--regcoe", type=float, default=0.5,
                    help='Regularization Coefficient when not replaying')
parser.add_argument("--regcoe_rply", type=float, default=5.0,
                    help='Regularization Coefficient when replaying')
parser.add_argument("--reggen", type=float, default=0.5,
                    help='Regularization Coefficient on General Space')
parser.add_argument("--regspe", type=float, default=0.5,
                    help='Regularization Coefficient on Specific Space')
parser.add_argument("--store_ratio", type=float, default=0.01,
                    help='how many samples to store for replaying')
parser.add_argument('--tasks', nargs='+', type=str,
                    default=['ag', 'yelp', 'amazon', 'yahoo', 'dbpedia'], help='Task Sequence')
parser.add_argument('--select_best', nargs='+', type=bool,
                    default=[True, True, True, True, True],
                    help='whether picking the model with best val acc on each task')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW, get_constant_schedule_with_warmup
from sklearn.cluster import KMeans, MiniBatchKMeans

from model import Model, Predictor
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
            for i in range(length):
                x = torch.tensor(self.examples[i]).view(1, -1).to(device)
                mask = torch.tensor(self.masks[i]).view(1, -1).to(device)
                g_fea, s_fea, _, _, _ = model(x, mask)
                fea = torch.cat([g_fea, s_fea], dim=1).view(-1).data.cpu().numpy()
                self.features.append(fea)
        print(len(self.features))
        print(len(self.labels))

    def get_random_batch(self, batch_size, task_id=None):
        if task_id is None:
            permutations = np.random.permutation(len(self.labels))
            index = permutations[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
        else:
            index = [i for i in range(len(self.labels)) if self.tasks[i] == task_id]
            np.random.shuffle(index)
            index = index[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
        return torch.tensor(mini_examples), torch.tensor(mini_masks), torch.tensor(mini_labels), \
               torch.tensor(mini_tasks), torch.tensor(mini_features)

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


def random_seq(src):
    #adding [SEP] to unify the format of samples for NSP
    batch_size = src.size(0)
    length = src.size(1)
    dst = []
    for i in range(batch_size):
        cur = src[i]
        first_pad = (cur.tolist() + [0]).index(0)
        cur = cur[1:first_pad].tolist()
        cur = random_string(cur)
        padding = [0] * (length - len(cur) - 1)
        dst.append(torch.tensor([101] + cur + padding))
    return torch.stack(dst).to(device)


def random_string(str):
    #randomly split positive samples into two halves and add [SEP] between them
    str.remove(102)
    str.remove(102)

    len1 = len(str)
    if len1 == 1:
        cut = 1
    else:
        cut = np.random.randint(1, len1)
    str = str[:cut] + [102] + str[cut:] + [102]
    return str


def change_string(str):
    #creating negative samples for NSP by randomly splitting positive samples
    #and swapping two halves
    str.remove(102)
    str.remove(102)

    len1 = len(str)
    if len1 == 1:
        cut = 1
    else:
        cut = np.random.randint(1, len1)
    str = str[cut:] + [102] + str[:cut] + [102]
    return str


def get_permutation_batch(src, src_mask):
    #create negative samples for Next Sentence Prediction
    batch_size = src.size(0)
    length = src.size(1)
    dst = []
    dst_mask = []
    lbl = []
    for i in range(batch_size):
        cur = src[i]
        mask = src_mask[i].tolist()
        first_pad = (cur.tolist() + [0]).index(0)
        cur = cur[1:first_pad].tolist()
        cur = change_string(cur)
        lbl.append(1)

        padding = [0] * (length - len(cur) - 1)
        dst.append(torch.tensor([101] + cur + padding))
        dst_mask.append(torch.tensor(mask))
    return torch.stack(dst).to(device), torch.stack(dst_mask).to(device), torch.tensor(lbl).to(device)


def train_step(model, optimizer, nsp_CR, cls_CR, x, mask, y, t, task_id, replay,
               x_feature, predictor, optimizer_P, scheduler, scheduler_P):
    batch_size = x.size(0)

    model.train()
    predictor.train()
    model.zero_grad()
    predictor.zero_grad()

    x = random_seq(x)
    pre_lbl = None

    # If Next Sentence Prediction is added, augment the training data with permuted data
    if args.disen:
        p_x, p_mask, p_lbl = get_permutation_batch(x, mask)
        x = torch.cat([x, p_x], dim=0)
        mask = torch.cat([mask, p_mask], dim=0)
        r_lbl = torch.zeros_like(p_lbl)
        nsp_lbl = torch.cat([r_lbl, p_lbl], dim=0)

        y = torch.cat([y, y], dim=0)
        t = torch.cat([t, t], dim=0)

    total_g_fea, total_s_fea, cls_pred, task_pred, _ = model(x, mask)

    if args.disen:
        g_fea = total_g_fea[:batch_size, :]
        s_fea = total_s_fea[:batch_size, :]
    else:
        g_fea = total_g_fea
        s_fea = total_s_fea

    # Calculate classification loss
    _, pred_cls = cls_pred.max(1)
    correct_cls = pred_cls.eq(y.view_as(pred_cls)).sum().item()
    cls_loss = cls_CR(cls_pred, y)

    task_loss = torch.tensor(0.0).to(device)
    reg_loss = torch.tensor(0.0).to(device)
    nsp_loss = torch.tensor(0.0).to(device)

    # Calculate regularization loss
    if x_feature is not None and args.reg is True:
        fea_len = g_fea.size(1)
        g_fea = g_fea[:batch_size, :]
        s_fea = s_fea[:batch_size, :]
        old_g_fea = x_feature[:, :fea_len]
        old_s_fea = x_feature[:, fea_len:]

        reg_loss += args.regspe * torch.nn.functional.mse_loss(s_fea, old_s_fea) + \
                    args.reggen * torch.nn.functional.mse_loss(g_fea, old_g_fea)

        if replay and task_id > 0:
            reg_loss *= args.regcoe_rply
        elif not replay and task_id > 0:
            reg_loss *= args.regcoe
        elif task_id == 0:
            reg_loss *= 0.0  #no reg loss on the 1st task

    # Calculate task loss only when in replay batch
    task_pred = task_pred[:, :task_id + 1]
    _, pred_task = task_pred.max(1)
    correct_task = pred_task.eq(t.view_as(pred_task)).sum().item()
    if task_id > 0 and replay:
        task_loss += args.tskcoe * cls_CR(task_pred, t)

    # Calculate Next Sentence Prediction loss
    nsp_acc = 0.0
    if args.disen:
        nsp_output = predictor(total_g_fea)
        nsp_loss += args.nspcoe * nsp_CR(nsp_output, nsp_lbl)

        _, nsp_pred = nsp_output.max(1)
        nsp_correct = nsp_pred.eq(nsp_lbl.view_as(nsp_pred)).sum().item()
        nsp_acc = nsp_correct * 1.0 / (batch_size * 2.0)

    loss = cls_loss + task_loss + reg_loss + nsp_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    if args.disen:
        optimizer_P.step()
        scheduler_P.step()

    return nsp_acc, correct_cls, correct_task, nsp_loss.item(), task_loss.item(), cls_loss.item(), reg_loss.item()


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


def select_samples_to_store(model, buffer, data_loader, task_id):
    ### ----------- add examples to memory ------------------ ##
    x_list = []
    mask_list = []
    y_list = []
    fea_list = []

    model.eval()
    with torch.no_grad():
        for x, mask, y in data_loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            _, _, _, _, bert_emb = model(x, mask)
            x_list.append(x.to("cpu"))
            mask_list.append(mask.to("cpu"))
            y_list.append(y.to("cpu"))
            # Kmeans on bert embedding
            fea_list.append(bert_emb.to("cpu"))
    x_list = torch.cat(x_list, dim=0).data.cpu().numpy()
    mask_list = torch.cat(mask_list, dim=0).data.cpu().numpy()
    y_list = torch.cat(y_list, dim=0).data.cpu().numpy()
    fea_list = torch.cat(fea_list, dim=0).data.cpu().numpy()

    # if use KMeans
    if args.kmeans:
        n_clu = int(args.store_ratio * len(x_list))
        estimator = KMeans(n_clusters=n_clu, random_state=args.seed)
        estimator.fit(fea_list)
        label_pred = estimator.labels_
        centroids = estimator.cluster_centers_
        for clu_id in range(n_clu):
            index = [i for i in range(len(label_pred)) if label_pred[i] == clu_id]
            closest = float("inf")
            closest_x = None
            closest_mask = None
            closest_y = None
            for j in index:
                dis = np.sqrt(np.sum(np.square(centroids[clu_id] - fea_list[j])))
                if dis < closest:
                    closest_x = x_list[j]
                    closest_mask = mask_list[j]
                    closest_y = y_list[j]
                    closest = dis

            if closest_x is not None:
                buffer.append(closest_x, closest_mask, closest_y, task_id)
    else:
        permutations = np.random.permutation(len(x_list))
        index = permutations[:int(args.store_ratio * len(x_list))]
        for j in index:
            buffer.append(x_list[j], mask_list[j], y_list[j], task_id)
    print("Buffer size:{}".format(len(buffer)))
    print(buffer.labels)
    b_lbl = np.unique(buffer.labels)
    for i in b_lbl:
        print("Label {} in Buffer: {}".format(i, buffer.labels.count(i)))


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

    buffer = Memory()
    model = Model(
        n_tasks=task_num,
        n_class=total_classes,
        hidden_size=args.hidden_size).to(args.device)

    predictor = Predictor(2, hidden_size=args.hidden_size).to(args.device)
    nsp_CR = torch.nn.CrossEntropyLoss()
    cls_CR = torch.nn.CrossEntropyLoss()

    for task_id in range(task_num):
        data_loader = train_loaders[task_id]
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

        scheduler = get_constant_schedule_with_warmup(optimizer, 1000)
        scheduler_P = get_constant_schedule_with_warmup(optimizer_P, 1000)

        best_acc = 0
        best_model = deepcopy(model.state_dict())
        best_predictor = deepcopy(predictor.state_dict())

        #store the features outputted by original model
        buffer.store_features(model)
        acc_track = []

        currentBuffer = Memory()
        model.eval()
        print("INIT current buffer...")
        with torch.no_grad():
            for x, mask, y in data_loader:
                for i, yi in enumerate(y):
                    currentBuffer.append(x[i].data.cpu().numpy(), mask[i].data.cpu().numpy(), y[i].item(), task_id)
        print("Start Storing Features...")
        currentBuffer.store_features(model)

        length = len(currentBuffer)

        for epoch in range(args.epochs[task_id]):
            # Training Loss/Accuracy on replaying batches
            cls_losses = []
            reg_losses = []
            nsp_losses = []
            tsk_accs = []
            cls_accs = []
            nsp_accs = []

            #Training Loss/Accuracy on batches of current task
            current_cls_losses = []
            current_reg_losses = []
            current_nsp_losses = []
            current_tsk_accs = []
            current_cls_accs = []
            current_nsp_accs = []

            iteration = 1
            for x, mask, y, t, origin_fea in tqdm(currentBuffer.get_minibatch(args.batch_size),
                                                  total=length // args.batch_size, ncols=100):
                if iteration % args.replay_freq == 0 and task_id > 0:
                    total_x, total_mask, total_y, total_t, total_fea = x, mask, y, t, origin_fea
                    for j in range(task_id):
                        old_x, old_mask, old_y, old_t, old_fea = \
                            buffer.get_random_batch(args.batch_size, j)
                        total_x = torch.cat([old_x, total_x], dim=0)
                        total_mask = torch.cat([old_mask, total_mask], dim=0)
                        total_y = torch.cat([old_y, total_y], dim=0)
                        total_t = torch.cat([old_t, total_t], dim=0)
                        total_fea = torch.cat([old_fea, total_fea], dim=0)
                    permutation = np.random.permutation(total_x.shape[0])
                    total_x = total_x[permutation, :]
                    total_mask = total_mask[permutation, :]
                    total_y = total_y[permutation]
                    total_t = total_t[permutation]
                    total_fea = total_fea[permutation, :]
                    for j in range(task_id + 1):
                        x = total_x[j * args.batch_size: (j + 1) * args.batch_size, :]
                        mask = total_mask[j * args.batch_size: (j + 1) * args.batch_size, :]
                        y = total_y[j * args.batch_size: (j + 1) * args.batch_size]
                        t = total_t[j * args.batch_size: (j + 1) * args.batch_size]
                        fea = total_fea[j * args.batch_size: (j + 1) * args.batch_size, :]
                        x, mask, y, t, fea = \
                            x.to(device), mask.to(device), y.to(device), t.to(device), fea.to(device)

                        nsp_acc, correct_cls, correct_task, nsp_loss, task_loss, cls_loss, reg_loss, = \
                            train_step(model, optimizer, nsp_CR, cls_CR, x, mask, y, t, task_id, True,
                                       fea, predictor, optimizer_P, scheduler, scheduler_P)

                        cls_losses.append(cls_loss)
                        reg_losses.append(reg_loss)
                        nsp_losses.append(nsp_loss)

                        tsk_accs.append(correct_task * 0.5 / x.size(0))
                        cls_accs.append(correct_cls * 0.5 / x.size(0))
                        nsp_accs.append(nsp_acc)

                else:
                    x, mask, y, t, origin_fea = x.to(device), mask.to(device), y.to(device), t.to(
                        device), origin_fea.to(device)
                    pre_acc, correct_cls, correct_task, pre_loss, task_loss, cls_loss, reg_loss = \
                        train_step(model, optimizer, nsp_CR, cls_CR, x, mask, y, t, task_id, False,
                                   origin_fea, predictor, optimizer_P, scheduler, scheduler_P)

                    current_cls_losses.append(cls_loss)
                    current_reg_losses.append(reg_loss)
                    current_nsp_losses.append(pre_loss)

                    current_tsk_accs.append(correct_task * 1.0 / x.size(0))
                    current_cls_accs.append(correct_cls * 1.0 / x.size(0))
                    current_nsp_accs.append(pre_acc)

                if iteration % 250 == 0:
                    print("----------------Validation-----------------")
                    avg_acc, acc_list = validation(model, task_id, validation_loaders)
                    acc_track.append(acc_list)

                    if avg_acc > best_acc:
                        print("------------------Best Model Till Now------------------------")
                        best_acc = avg_acc
                        best_model = deepcopy(model.state_dict())
                        best_predictor = deepcopy(predictor.state_dict())

                iteration += 1

            if len(reg_losses) > 0:
                print("Mean REG Loss: {}".format(np.mean(reg_losses)))
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
            if len(current_reg_losses) > 0:
                print("Mean Current REG Loss: {}".format(np.mean(current_reg_losses)))
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

        if args.select_best[task_id]:
            model.load_state_dict(deepcopy(best_model))
            predictor.load_state_dict(deepcopy(best_predictor))
        print("------------------Best Result------------------")
        avg_acc, _ = validation(model, task_id, test_loaders)
        print("Best avg acc: {}".format(avg_acc))

        if args.dump is True:
            task_order = '_'.join(args.tasks)
            path = './dump/' + task_order + '_' + str(args.seed) + '_' + str(task_id) + '.pt'
            torch.save(model, path)

        select_samples_to_store(model, buffer, data_loader, task_id)


if __name__ == '__main__':
    print(args)
    main()
