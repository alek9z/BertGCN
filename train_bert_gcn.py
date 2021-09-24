import argparse
import logging
import os
import shutil

import dgl
import torch as th
import torch.nn.functional as F
import torch.utils.data as Data
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, ClassificationReport
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import lr_scheduler

from model import BertGCN, BertGAT
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=512, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased',
                             'bert-base-cased', 'dbmdz/bert-base-italian-xxl-cased', 'flaubert/flaubert_base_cased',
                             'xlm-roberta-base'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--dataset', default='enwiki', choices=["enwiki", "itwiki", "frwiki", "rcv1en", "rcv2it", "rcv2fr"])
parser.add_argument('--checkpoint_dir', default=None,
                    help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=200,
                    help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
parser.add_argument('--heads', type=int, default=8, help='the number of attention heads for gat')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--bert_lr', type=float, default=1e-5)

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:0')

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
# Model


# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
'''

# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# instantiate model according to class number
if gcn_model == 'gcn':
    model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout)
else:
    model = BertGAT(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    heads=heads, n_hidden=n_hidden, dropout=dropout)

if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
    # Workaround to load custom pretrained classifier, with a different checkpoint structure
    model.bert_clf.load_state_dict({".".join(k.split(".")[1:]): v for k, v in ckpt['model'].items()})

# load documents and compute input encodings
corpse_file = './data/corpus/' + dataset + '_shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    #     print(input.keys())
    return input.input_ids, input.attention_mask


input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat(
    [attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val

# document mask used for update feature
doc_mask = train_mask + val_mask + test_mask

# build DGL Graph
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

# create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)


# Training
def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batch size to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_clf(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True) \
                         .hidden_states[-1][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


optimizer = th.optim.Adam([
    {'params': model.bert_clf.parameters(), 'lr': bert_lr},
    {'params': model.gcn.parameters(), 'lr': gcn_lr},
], lr=1e-3)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx,) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask].float()
    y_true = g.ndata['label_train'][idx][train_mask].float()
    # Must be 2 floats for BCE
    loss = F.binary_cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.long().detach().cpu()
            assert y_true.bool().any(dim=1).all().item() is True
            y_pred = th.round(y_pred).long().detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
            pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        else:
            train_acc, pre, rec, f1 = 1, 1, 1, 1  # nothing to predict
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx,) = [x.to(gpu) for x in batch]
        # NOTE: must be 2 floats
        y_pred = th.round(model(g, idx)).float()
        y_true = g.ndata['label'][idx].float()
        return y_pred, y_true


evaluator = Engine(test_step)
metrics = {
    'acc': Accuracy(is_multilabel=True),
    'cr': ClassificationReport(output_dict=True, is_multilabel=True),
    'nll': Loss(th.nn.BCELoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics_res = evaluator.state.metrics
    train_acc, train_nll, train_cr = metrics_res["acc"], metrics_res["nll"], metrics_res["cr"]["macro avg"]
    train_pre, train_rec, train_f1 = train_cr["precision"], train_cr["recall"], train_cr["f1-score"]

    evaluator.run(idx_loader_val)
    metrics_res = evaluator.state.metrics
    val_acc, val_nll, val_cr = metrics_res["acc"], metrics_res["nll"], metrics_res["cr"]["macro avg"]
    val_pre, val_rec, val_f1 = val_cr["precision"], val_cr["recall"], val_cr["f1-score"]

    evaluator.run(idx_loader_test)
    metrics_res = evaluator.state.metrics
    test_acc, test_nll, test_cr = metrics_res["acc"], metrics_res["nll"], metrics_res["cr"]["macro avg"]
    test_pre, test_rec, test_f1 = test_cr["precision"], test_cr["recall"], test_cr["f1-score"]

    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f} pre: {:.4f} rec: {:.4f} f1: {:.4f} || Val acc: {:.4f} loss: {:.4f} "
        "pre: {:.4f} rec: {:.4f} f1: {:.4f} || Test acc: {:.4f} loss: {:.4f} pre: {:.4f} rec: {:.4f} f1: {:.4f}"
            .format(trainer.state.epoch, train_acc, train_nll, train_pre, train_rec, train_f1, val_acc, val_nll,
                    val_pre, val_rec, val_f1, test_acc, test_nll, test_pre, test_rec, test_f1)
    )
    if val_f1 > log_training_results.best_metric:
        logger.info("New checkpoint")
        th.save(
            {
                'model': model.bert_clf.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_metric = val_f1


log_training_results.best_metric = -1.0
g = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)
