# %%
# Imports
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN,DCRNN,GConvGRU

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
import torch.nn.functional as F
from pathlib import Path
import pathlib
import numpy as np
import torch
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric_temporal.nn.recurrent import AGCRN


# %%
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("Device is:",device)

# %%
# traffic_dataset = torch.load('./data/graph_data/1_hr_time_window_dynamic_graph/dataset_all_route_dynamic_graph_7_features.pt')
# traffic_dataset = torch.load('./data/graph_data/1_hr_time_window_dynamic_graph/dataset_all_route_dynamic_graph.pt')
# traffic_dataset = torch.load('./data/graph_data/1_hr_time_window_static_graph/dataset_all_route_single_graph.pt')
traffic_dataset = torch.load('../SNA_create_graph/data/graph_data/1_hr_time_window_static_graph/dataset_all_route_single_graph_all_features.pt')

# %%
# # Removing graphs that just have zero bin in the graph to reduce sparsity
# cleaned_traffic_dataset = []

# for dataset in traffic_dataset:
#     y_list = np.array(dataset.y)
#     non_zero = np.count_nonzero(y_list)
#     if(non_zero == 0):
#         continue
#     else:
#         cleaned_traffic_dataset.append(dataset)

# traffic_dataset = cleaned_traffic_dataset

# %%
len(traffic_dataset)

# %%
traffic_dataset[0].y.shape[0]

# %%
for data in traffic_dataset:
    if data.y.shape[0] != 757:
        print(data)
        break

# %%



Edge_Indices = Sequence[Union[np.ndarray, None]]
Edge_Weights = Sequence[Union[np.ndarray, None]]
Node_Features = Sequence[Union[np.ndarray, None]]
Targets = Sequence[Union[np.ndarray, None]]
Additional_Features = Sequence[np.ndarray]


class DynamicGraphTemporalSignal_custom(object):
    r"""A data iterator object to contain a dynamic graph with a
    changing edge set and weights . The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric Data object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.

    Args:
        edge_indices (Sequence of Numpy arrays): Sequence of edge index tensors.
        edge_weights (Sequence of Numpy arrays): Sequence of edge weight tensors.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """

    def __init__(
        self,
        edge_indices: Edge_Indices,
        edge_weights: Edge_Weights,
        features: Node_Features,
        targets: Targets,
        **kwargs: Additional_Features
    ):
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.features = features
        self.targets = targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        assert len(self.edge_indices) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        assert len(self.features) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self, time_index: int):
        if self.edge_indices[time_index] is None:
            return self.edge_indices[time_index]
        else:
            return torch.LongTensor(self.edge_indices[time_index])

    def _get_edge_weight(self, time_index: int):
        if self.edge_weights[time_index] is None:
            return self.edge_weights[time_index]
        else:
            return torch.FloatTensor(self.edge_weights[time_index])

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            return torch.LongTensor(self.targets[time_index])
            # if self.targets[time_index].dtype.kind == "i":
            #     return torch.LongTensor(self.targets[time_index])
            # elif self.targets[time_index].dtype.kind == "f":
            #     return torch.FloatTensor(self.targets[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = DynamicGraphTemporalSignal(
                self.edge_indices[time_index],
                self.edge_weights[time_index],
                self.features[time_index],
                self.targets[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index(time_index)
            edge_weight = self._get_edge_weight(time_index)
            y = self._get_target(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, **additional_features)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self
    
class myNewOwnLoader(object):
    """A dataset of county level chicken pox cases in Hungary between 2004
    and 2014. We made it public during the development of PyTorch Geometric
    Temporal. The underlying graph is static - vertices are counties and
    edges are neighbourhoods. Vertex features are lagged weekly counts of the
    chickenpox cases (we included 4 lags). The target is the weekly number of
    cases for the upcoming week (signed integers). Our dataset consist of more
    than 500 snapshots (weeks).
    """

    def __init__(self, dataset):
        # self._read_web_data()
        self._get_dataset(dataset)

    def _get_dataset(self,dataset):
        self._dataset = dataset

    def _get_edges(self):
        # self._edges = np.array(self._dataset["edges"]).T
        # ei = []
        # for data in self._dataset:
        #     ei.append(data.edge_index)
        e = []
        for data in self._dataset:
            e.append(data.edge_index)

        self._edges = e


    def _get_edge_weights(self):
        # self._edge_weights = np.ones(self._edges.shape[1])
        # ew = []
        # for data in self._dataset:
        #     ew.append(data.edge_attr)
        ew = []
        for data in self._dataset:
            ew.append(data.edge_attr[:, 0])
            # ew.append(data.edge_attr)


        self._edge_weights = ew

        # self._edge_weights = np.array(self._dataset[0].edge_attr)
        # self._edge_weights = self._dataset.edge_attr

    def _get_targets(self):

        y_class = []
        for data in self._dataset:
            y_class.append(data.y)

        self.targets = y_class
        # self.targets = self._dataset.y

    def _get_features(self):
        f = []
        for data in self._dataset:
            f.append(data.x)

        self.features = f
        # self.features = self._dataset.x

    def get_dataset(self, lags: int = 8) -> DynamicGraphTemporalSignal_custom:
        """Returning the Chickenpox Hungary data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets()
        self._get_features()
        # self._get_targets_and_features()
        # self.

        dataset = DynamicGraphTemporalSignal_custom(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

# %%
test_size = int(len(traffic_dataset)*0.2)
val_size = int(len(traffic_dataset)*0.1)
train_size = len(traffic_dataset) - test_size - val_size

train_dataset = traffic_dataset[:train_size]
validation_dataset = traffic_dataset[train_size+1:train_size+val_size]
test_dataset = traffic_dataset[train_size+val_size+1:train_size+val_size+test_size]

print(len(train_dataset), len(validation_dataset), len(test_dataset))


# %%
def get_class_counts(dataset):
    y_train = []

    for dataset in dataset:
        y_train += list(np.array(dataset.y))

    y_train = np.array(y_train)
    class_counts = np.bincount(y_train)
    return class_counts, len(y_train)

train_count, total_train = get_class_counts(train_dataset)
val_count, total_val = get_class_counts(validation_dataset)
test_count, total_test = get_class_counts(test_dataset)

# Getting the percentage of bins to see how the distribution looks like
print(train_count*100/total_train)
print(val_count*100/total_val)
print(test_count*100/total_test)
# I can say its pretty similar but very very sparse

# %%
newloader_train = myNewOwnLoader(train_dataset)
train_dataset = newloader_train.get_dataset(lags=8)

newloader_val = myNewOwnLoader(validation_dataset)
validation_dataset = newloader_val.get_dataset(lags=8)

newloader_test = myNewOwnLoader(test_dataset)
test_dataset = newloader_test.get_dataset(lags=8)

# %%
train_dataset[0], validation_dataset[0], test_dataset[0]

# %%
train_dataset[0].edge_attr

# %%
y_train = []

for dataset in train_dataset:
    y_train += list(np.array(dataset.y))
    # y_train.append(np.array(dataset.y))



# %%
y_train = np.array(y_train)

class_counts = np.bincount(y_train)
num_classes = 4
total_samples = len(y_train)

class_weights = []
for count in class_counts:
    weight = 1 / (count / total_samples)
    class_weights.append(weight)

# %%
class_counts, class_weights

# array([158929,  86989,  21581,   8111])

# %%

def train_one_epoch(train_loader, e):
    model.train()

    # Initial parameters
    cost = 0
    h = None
    count = 0 
    correct = 0
    for time, snapshot in enumerate(train_loader):
        snapshot.to(device)
        # x = snapshot.x.view(1, no_of_nodes, no_of_node_features)
        # y_hat, h = model(x, e, h)
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

        # Getting the focal loss
        cost = cost + focal_loss(y_hat.squeeze(), snapshot.y.long())
        pred = y_hat.argmax(dim=1)  # Use the class with highest probability.
        yp = np.array(pred.cpu()).flatten()
        yt = np.array(snapshot.y.cpu()).flatten()
        # Find the indices where yt is equal to 4
        indices_to_remove = np.where(yt == 4)
        yp = np.delete(yp, indices_to_remove)
        yt = np.delete(yt, indices_to_remove)
        correct += int((yp == yt).sum()) 
        count += len(yt)
    cost = cost / (time+1) #Getting the mean loss
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    accuracy = correct / count

    return accuracy, cost

def validate_one_epoch(validation_loader, e):
    model.eval()

    # Initial parameters
    cost = 0
    h = None
    count = 0 
    correct = 0
    for time, snapshot in enumerate(validation_loader):
        snapshot.to(device)
        # x = snapshot.x.view(1, no_of_nodes, no_of_node_features)
        # y_hat, h = model(x, e, h)
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

        # Getting the focal loss
        cost = cost + focal_loss(y_hat.squeeze(), snapshot.y.long())
        pred = y_hat.argmax(dim=1)  # Use the class with highest probability.
        yp = np.array(pred.cpu()).flatten()
        yt = np.array(snapshot.y.cpu()).flatten()
        # Find the indices where yt is equal to 4
        indices_to_remove = np.where(yt == 4)
        yp = np.delete(yp, indices_to_remove)
        yt = np.delete(yt, indices_to_remove)

        correct += int((yp == yt).sum()) 
        count += len(yt)
    cost = cost / (time+1) #Getting the mean loss
    
    accuracy = correct / count
    return accuracy, cost


# %%


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# %%
# Data(x=[45, 10], edge_index=[2, 94], edge_attr=[94, 7], y=[45])

# %%
# no_of_nodes = 45
no_of_node_features = 10
num_classes = 5
lr = 0.01
edge_features = 1

EPOCHS = 400
patience = 150
early_stopper = EarlyStopper(patience=patience, min_delta=0)

# class RecurrentGCN(torch.nn.Module):
#     def __init__(self, node_features, num_classes):
#         super(RecurrentGCN, self).__init__()
#         self.transform = torch.nn.Linear(node_features,32)

#         self.recurrent = GConvGRU(in_channels = 32,
#                               out_channels = 32,K=3)
        
#         self.linear1 = torch.nn.Linear(node_features, 32)


#         # Future work add some MLP layers, increase out channels - to 64.in the future
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(32+node_features, 32),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.5),
#             torch.nn.Linear(32, 16),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.5),
#             torch.nn.Linear(16,num_classes)
#         )
#         # torch.nn.Linear(node_features, 32)
#         # self.linear2 = torch.nn.Linear(32+node_features, 32)
#         # self.linear = torch.nn.Linear(32, num_classes)


#     def forward(self, x, edge_index, edge_weight, prev_hidden_state):
        
#         x = self.transform(x)
#         # Learning the edge attributes using MLP layer before sending in so it can be of dimension (no_of_features X 1)
#         input_feat = x.clone()
#         prev_hidden_state = self.recurrent(x, edge_index, edge_weight, prev_hidden_state).relu()
#         # prev_hidden_state = F.relu(prev_hidden_state)
        
#         h = F.relu(self.linear1(prev_hidden_state))
#         h = F.dropout(h,0.5)

#         combined = torch.cat((input_feat,h),dim = 1)

#         h = self.mlp(combined)

#         # h = F.relu(self.linear2(combined))
#         # h = self.linear(h)
        
#         h = F.log_softmax(h,dim=-1)
#         return h, prev_hidden_state

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(in_channels = node_features,
                              out_channels = node_features,K=3)
        # Future work add some MLP layers, increase out channels - to 64.in the future
        self.linear1 = torch.nn.Linear(node_features, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)

        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        h = self.linear(h)
        h = F.log_softmax(h,dim=-1)
        return h
        
model = RecurrentGCN(node_features = no_of_node_features, num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# %%
class_weights = torch.FloatTensor(class_weights).to(device)
focal_loss = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
	model='FocalLoss',
	alpha=class_weights,
	gamma=2,
	reduction='mean',
	force_reload=False
).to(device)

# %%

model_checkpoint_path = './model_checkpoint/GConvGRU/'
pathlib.Path(model_checkpoint_path).mkdir(parents=True, exist_ok=True) 

e = None
# e = torch.empty(no_of_nodes, 4)
# torch.nn.init.xavier_uniform_(e)

history = {
    'epoch':[],
    'train_acc':[],
    'validation_acc':[],
    'train_loss':[],
    'validation_loss':[]
}

for epoch in tqdm(range(EPOCHS)):
    train_acc, train_loss = train_one_epoch(train_dataset, e)
    validation_acc, validation_loss = validate_one_epoch(validation_dataset, e)
    # print(validation_loss)
    PATH  = f'{model_checkpoint_path}/model_GConvGRU_{epoch}_static_graph.pt'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            }, PATH)
    
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {validation_acc:.4f}, Val Loss: {validation_loss:.4f}')

    history['epoch'].append(epoch+1)
    history['train_acc'].append(train_acc)
    history['validation_acc'].append(validation_acc)
    history['train_loss'].append(float(train_loss.data))
    history['validation_loss'].append(float(validation_loss.data))
    
    if early_stopper.early_stop(validation_loss):     
        print(f"Stopping early! {validation_loss}, {epoch}")      
        break


# %%
# Experiment_results_saving path

experiment_results_path = './exp_results/'
pathlib.Path(experiment_results_path).mkdir(parents=True, exist_ok=True)

# %%
# best_epoch = epoch - patience
def find_min_val(val_list):
    res = 0
    cur_min = val_list[0]
    for i in range(1,len(val_list)):
        if val_list[i] <= cur_min:
            res = i
            cur_min = val_list[i]
    return res

best_iteration = find_min_val(history['validation_loss'])
best_epoch = history['epoch'][best_iteration]-1



# %%
# tmodel = torch.load(f'/home/sgupta/WORK/Stoplevel_day_ahead_prediction/GNN_new_graph_stoplevel/model_checkpoint/GConvGRU/model_GConvGRU_{best_epoch}_static_graph.pt')

model2 = model
# model2.load_state_dict(tmodel['model_state_dict'])

# %%
lr = lr
EPOCHS = best_epoch
patience = patience
loss = 'focal_loss'
node_features_used = no_of_node_features
edge_features_used = 1

# %%
curr_experiment_path = f'{experiment_results_path}/GConvGRU_1_hr_time_window/lr_{lr}_epochs_{EPOCHS}_patience_{patience}_loss_{loss}_no_node_f_{node_features_used}_no_edge_f_{edge_features_used}_static_graph'
pathlib.Path(curr_experiment_path).mkdir(parents=True, exist_ok=True) 

# %%

def test(loader):
    model.eval()

    correct=0
    count=0
    y_true = []
    y_pred = []

    h = None
    for time, snapshot in enumerate(loader):
        snapshot.to(device)
        # x = snapshot.x.view(1, no_of_nodes,no_of_node_features)
        y_hat = model2(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        
        # Assuming snapshot.y is of shape [20] and has values in the range [0, 4]
        # snapshot_y_one_hot = F.one_hot(snapshot.y, num_classes=num_classes).unsqueeze(0)

        pred = y_hat.argmax(dim=1)  # Use the class with highest probability.
        y_true.append(snapshot.y)
        y_pred.append(pred)

        yp = np.array(pred.cpu()).flatten()
        yt = np.array(snapshot.y.cpu()).flatten()
        correct += int((yp == yt).sum()) 
        count += len(yt)

    return (correct/count), y_true, y_pred


test_acc,  y_true, y_pred = test(test_dataset)

# %%
yt = np.concatenate([tensor.flatten().cpu() for tensor in y_true])
yp = np.concatenate([tensor.flatten().cpu() for tensor in y_pred])

temp = pd.DataFrame(columns=['ytrue','ypred'])
temp.ytrue = yt
temp.ypred = yp

# %%



def get_CM(y_true,y_pred,title,comment,path):
    test_acc = (y_pred == y_true).sum()/len(y_true)

    print(f"Accuracy {comment}:",test_acc)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    _ = disp.plot(ax=ax)
    # Add a title to the plot
    ax.set_title(title + " - Real Count")
    plt.savefig(f'{path}/CM_real_55_{comment}.jpg')
    # plt.show()

    # Getting the percentage CM
    cm = ((cm * 100) / (cm.sum(axis=1)[:, np.newaxis]))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    _ = disp.plot(ax=ax)
    # Add a title to the plot
    ax.set_title(title + " - Percentages")
    plt.savefig(f'{path}/CM_percentage_55{comment}.jpg')
    # plt.show()

    return test_acc

# %%
test_acc_w_4th = get_CM(temp.ytrue,temp.ypred,'ROUTE 55 ','with 4th bin',curr_experiment_path)
history['test_acc_with_4th_bin'] = test_acc_w_4th


# %%
temp2 = temp
temp2 = temp2[temp2.ytrue != 4]
temp2.reset_index(drop=True,inplace=True)
test_acc_w_o_4th = get_CM(temp2.ytrue,temp2.ypred,'ROUTE 55 no 4th bin','without 4th bin',curr_experiment_path)
history['test_acc_without_4th_bin'] = test_acc_w_o_4th

# %%
def accuracy_metric(y_true,y_pred):
        a = []
        
        for i in range(len(y_true)):
            yt = y_true[i]
            yp = y_pred[i]

            temp = 1 - (np.abs(yt-yp)/num_classes)
            a.append(temp)
        # print(a)
        accuracy = sum(a)/len(y_true)

        return accuracy

custom_accuracy = accuracy_metric(temp2.ytrue,temp2.ypred)
print('Custom Accuracy Metric:',custom_accuracy)
history['custom_accuracy'] = custom_accuracy

# %%
print(history)

# %%
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1_score, support = precision_recall_fscore_support(temp.ytrue,temp.ypred, average='weighted')

# %%
history_df = pd.DataFrame(history)
history_df.to_csv(f'{curr_experiment_path}/history_w_results.csv')

# %%
# Open a file in write mode ('w' or 'a' for append)
with open('./results/sn_de_1_hr/static_node_dynamic_edge_1_hr_output.txt', 'w') as file:
    # Write data to the file
    file.write(f"The precision, recall and f1-score are: {precision}, {recall} and {f1_score}")
    file.write(f'Custom Accuracy Metric: {custom_accuracy}')

# %%
temp.to_csv('./results/sn_de_1_hr/static_node_dynamic_edge_1_hr_preds.csv')


