import os
import numpy as np
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
from sklearn.metrics import average_precision_score, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict


def loading_item_data(func,name):
    item = []
    with open('/media/ST-18T/yuntong/SuperEdgeGO/data_collect/'+func+'/'+name+'.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            item.append(line)
    return item


def get_label(func):
    if func == 'mf':
        mf_dict = {}
        mf_func = []
        mf_label_dict = {}
        with open('/media/ST-18T/yuntong/SuperEdgeGO/data_collect/GO_label.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'mf' in line:
                    if 'mf:' == line.strip('\n'):
                        continue
                    else:
                        mf_func_list = line[3:].strip().split('\t')
                        mf_dict[pdb_chain_uid] = mf_func_list
        with open('/media/ST-18T/yuntong/SuperEdgeGO/data_collect/mf/mf_label.txt','r') as f:
            for line in f:
                line = line.strip('\n')
                mf_func.append(line)
        for i in mf_dict.keys():
            label = np.zeros(len(mf_func))
            for j in mf_dict[i]:
                if j in mf_func:
                    index = mf_func.index(j)
                    label[index] = 1
            mf_label_dict[i] = label
        return mf_label_dict,len(mf_func)
    elif func == 'bp':
        bp_dict = {}
        bp_func = []
        bp_label_dict = {}
        with open('/media/ST-18T/yuntong/SuperEdgeGO/data_collect/GO_label.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'bp' in line:
                    if 'bp:' == line.strip('\n'):
                        continue
                    else:
                        bp_func_list = line[3:].strip().split('\t')
                        bp_dict[pdb_chain_uid] = bp_func_list
        with open('/media/ST-18T/yuntong/SuperEdgeGO/data_collect/bp/bp_label.txt','r') as f:
            for line in f:
                line = line.strip('\n')
                bp_func.append(line)
        for i in bp_dict.keys():
            label = np.zeros(len(bp_func))
            for j in bp_dict[i]:
                if j in bp_func:
                    index = bp_func.index(j)
                    label[index] = 1
            bp_label_dict[i] = label
        return bp_label_dict,len(bp_func)
    elif func == 'cc':
        cc_dict = {}
        cc_func = []
        cc_label_dict = {}
        with open('/media/ST-18T/yuntong/SuperEdgeGO/data_collect/GO_label.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'cc' in line:
                    if 'cc:' == line.strip('\n'):
                        continue
                    else:
                        cc_func_list = line[3:].strip().split('\t')
                        cc_dict[pdb_chain_uid] = cc_func_list
        with open('/media/ST-18T/yuntong/SuperEdgeGO/data_collect/cc/cc_label.txt', 'r') as f:
            for line in f:
                line = line.strip('\n')
                cc_func.append(line)
        for i in cc_dict.keys():
            label = np.zeros(len(cc_func))
            for j in cc_dict[i]:
                if j in cc_func:
                    index = cc_func.index(j)
                    label[index] = 1
            cc_label_dict[i] = label
        return cc_label_dict, len(cc_func)


class PFPDataset(InMemoryDataset):
    def __init__(self, model_feature=None, train_data_X=None, train_data_Y=None,root='/tmp', transform=None, pre_transform=None, device=None):
        super(PFPDataset, self).__init__(root, transform, pre_transform)
        #self.dir=dir
        self.device = device
        # self.model_feature = model_feature
        self.X_data_list = train_data_X
        self.Y_data_list = train_data_Y

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    def __len__(self):
        return int(len(self.X_data_list))

    def __getitem__(self, idx):
        contact_map_name = self.X_data_list[idx] + '.npy'
        contactmap_folder_path = '/media/ST-18T/yuntong/SuperEdgeGO/data_collect/con_map'
        contact_map_path = os.path.join(contactmap_folder_path, contact_map_name)
        contact_map = np.load(contact_map_path)

        label = self.Y_data_list[self.X_data_list[idx]]
        # seq = self.X_feature_matrix_list[self.X_data_list[idx]]
        # seq_bytes = bytes(seq,encoding='utf-8')
        # feature_matrix = embed_sequence(self.model_feature,seq_bytes,use_cuda=True,device = self.device)

        feature_name = self.X_data_list[idx] + '.npy'
        feature_dir = '/media/ST-18T/yuntong/SuperEdgeGO/data_collect/fea_esm2'
        feature_path = os.path.join(feature_dir, feature_name)
        feature_matrix = torch.Tensor(np.load(feature_path))

        GCNData_mol = DATA.Data(x=feature_matrix,
                                edge_index=torch.LongTensor(contact_map),
                                label=torch.FloatTensor([label]), uid=self.X_data_list[idx])

        return GCNData_mol

def collate(data_list):
    mol_data_list = data_list
    batchA = Batch.from_data_list(mol_data_list)
    #embedding = [data[1] for data in data_list]
    #embedding = torch.stack(embedding).squeeze(dim=1)

    return batchA

def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max

def evaluate_performance(y_test, y_score):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()

    perf["M-aupr"] = 0.0
    n = 0
    aupr_list = []
    num_pos_list = []
    for i in range(n_classes):
        num_pos = sum(y_test[:, i])
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            aupr_list.append(ap)
            num_pos_list.append(str(num_pos))
    perf["M-aupr"] /= n
    # Compute micro-averaged AUPR

    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())
    alpha = 3
    y_new_pred = np.zeros_like(y_test)
    for i in range(y_test.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha * [1])


    perf['F-max'] = calculate_fmax(y_score, y_test)

    #计算auc
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_score.ravel())
    perf['AUC'] = auc(fpr, tpr)

    return perf


def get_results(Y_test, y_score):

    perf = defaultdict(dict)
    perf['all'] = evaluate_performance(Y_test, y_score)

    return perf

