import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from .torch_gat import GAT
from .torch_gcn import GCN


class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        config = AutoConfig.from_pretrained(pretrained_model, num_labels=nb_class)
        self.bert_clf = AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=config)
        self.feat_dim = self.bert_clf.config.hidden_size
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers - 1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        bert_output = self.bert_clf(input_ids, attention_mask, output_hidden_states=True)
        if self.training:
            cls_feats = bert_output.hidden_states[-1][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        cls_logit = bert_output.logits
        cls_pred = th.sigmoid(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.sigmoid(gcn_logit)
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        # pred = th.log(pred) # remove to preserve [0, 1] range
        return pred


class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32,
                 dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        config = AutoConfig.from_pretrained(pretrained_model, num_labels=nb_class)
        self.bert_clf = AutoModelForSequenceClassification.from_pretrained(pretrained_model, config=config)
        self.feat_dim = self.bert_clf.config.hidden_size
        self.gcn = GAT(
            num_layers=gcn_layers - 1,
            in_dim=self.feat_dim,
            num_hidden=n_hidden,
            num_classes=nb_class,
            heads=[heads] * (gcn_layers - 1) + [1],
            activation=F.elu,
            feat_drop=dropout,
            attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        bert_output = self.bert_clf(input_ids, attention_mask, output_hidden_states=True)
        if self.training:
            cls_feats = bert_output.hidden_states[-1][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        cls_logit = bert_output.logits
        cls_pred = th.sigmoid(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = th.sigmoid(gcn_logit)
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        # pred = th.log(pred)
        return pred
