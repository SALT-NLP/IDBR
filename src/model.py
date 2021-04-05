import torch
import torch.nn as nn
from transformers import BertModel, BertForPreTraining
from transformers.modeling_bert import \
    BertLMPredictionHead, BertPredictionHeadTransform, BertPreTrainingHeads
from copy import deepcopy


def get_tokens_id(x):
    tokens_id = x.new_ones(x.shape)
    for i in range(tokens_id.size(0)):
        ids = x[i].tolist()
        first_seq = ids.index(102)
        for j in range(first_seq + 1):
            tokens_id[i, j] = 0
    return tokens_id


class Model(nn.Module):

    def __init__(self, n_tasks, n_class, hidden_size):
        super().__init__()

        self.n_class = n_class
        self.n_tasks = n_tasks

        self.Bert = BertModel.from_pretrained('bert-base-uncased')

        self.config = self.Bert.config
        self.vocab_size = self.config.vocab_size

        self.hidden_size = hidden_size

        self.General_Encoder = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.Tanh()
        )

        self.Specific_Encoder = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.Tanh()
        )

        self.cls_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, n_class)
        )

        self.task_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, n_tasks)
        )

    def forward(self, x, mask):

        tokens_ids = get_tokens_id(x)
        output = self.Bert(input_ids=x,
                           attention_mask=mask,
                           token_type_ids=tokens_ids,
                           return_dict=True)

        sequence_output = output.last_hidden_state
        bert_embedding = sequence_output[:, 0:1, :].squeeze(dim=1)

        general_features = self.General_Encoder(bert_embedding)
        specific_features = self.Specific_Encoder(bert_embedding)

        task_pred = self.task_classifier(specific_features)

        features = torch.cat([general_features, specific_features], dim=1)
        cls_pred = self.cls_classifier(features)

        return general_features, specific_features, \
               cls_pred, task_pred, bert_embedding


class Predictor(torch.nn.Module):
    def __init__(self, num_class, hidden_size):
        super(Predictor, self).__init__()

        self.num_class = num_class

        self.dis = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, self.num_class)
        )

    def forward(self, z):
        return self.dis(z)


class BaseModel(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.n_class = n_class
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, n_class)
        )

    def forward(self, x):
        x, _ = self.bert(x)
        x = torch.mean(x, 1)
        logits = self.classifier(x)
        return logits