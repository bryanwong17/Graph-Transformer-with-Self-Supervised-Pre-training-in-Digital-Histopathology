
from __future__ import absolute_import, division, print_function

import torch
from utils.metrics import ConfusionMatrix

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def collate(batch):
    image = [ b['image'] for b in batch ] # w, h
    label = [ b['label'] for b in batch ]
    id = [ b['id'] for b in batch ]
    adj_s = [ b['adj_s'] for b in batch ]
    return {'image': image, 'label': label, 'id': id, 'adj_s': adj_s}

# features, label, adj
# 1 batch
def preparefeatureLabel(batch_graph, batch_label, batch_adjs):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0

    # in one batch, get the maximum node num and transfer the label
    for i in range(batch_size):
        labels[i] = batch_label[i]
        max_node_num = max(max_node_num, batch_graph[i].shape[0])
    
    # max node num in 1 , one node is one patch
    masks = torch.zeros(batch_size, max_node_num)
    adjs =  torch.zeros(batch_size, max_node_num, max_node_num)
    batch_node_feat = torch.zeros(batch_size, max_node_num, 2048)

    for i in range(batch_size):
        # get N (number of patches in one WSI)
        cur_node_num =  batch_graph[i].shape[0]
        
        # batch node attribute feature
        tmp_node_fea = batch_graph[i]
        batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

        # batch adjs
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]
        
        # batch masks
        masks[i,0:cur_node_num] = 1  

    node_feat = batch_node_feat.cuda()
    labels = labels.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()

    return node_feat, labels, adjs, masks

class Trainer(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()
    
    def plot_cm(self):
        self.metrics.plotcm()

    def train(self, sample, model):
        # sample['image'] = features
        # 1 batch
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'])
        # in one batch
        # if using graph
        pred,labels,loss = model.forward(node_feat, labels, adjs, masks)
        # if not
        # pred,labels,loss = model.forward(node_feat, labels)

        return pred,labels,loss

class Evaluator(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)
    
    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        self.metrics.plotcm()

    def eval_test(self, sample, model, graphcam_flag=False):
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'])
        if not graphcam_flag:
            with torch.no_grad():
                # if using graph
                pred,labels,loss = model.forward(node_feat, labels, adjs, masks)
                # if not
                # pred,labels,loss = model.forward(node_feat, labels)
        else:
            torch.set_grad_enabled(True)
            #pred,labels,loss= model.forward(node_feat, labels, adjs, masks, graphcam_flag=graphcam_flag)
            pred,labels,loss= model.forward(node_feat, labels, graphcam_flag=graphcam_flag)
        return pred,labels,loss
        