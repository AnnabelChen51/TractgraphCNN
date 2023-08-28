import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool,global_max_pool
from pygcn.layers import GraphConvolution as PYGCNConv
from torch import einsum
from einops import rearrange

class CNN_1D(nn.Module):  #number of parameters 12363266
    """Model for human-activity-recognition."""
    def __init__(self, input_channel,input_len,num_classses):
        super().__init__()
        # 计算全连接层输入维度
        self.classifier_input_len = 64 * (input_len - 12)
        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_channel, 64, 5),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
        )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.classifier_input_len, 128),
            nn.BatchNorm1d(128, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classses),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out



class PointNet(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,num_classses=2):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(97024, 128)
        #self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(128, num_classses)
        self.dropout = nn.Dropout(p=0.3)
        self.bnf1 = nn.BatchNorm1d(128)
        #self.bnf2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        #print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.size())
        x = x.view(x.size(0), -1)
        #x = x.max(dim=-1, keepdim=False)[0]
        #print(x.size())
        x = F.relu(self.bnf1(self.fc1(x)))
        #x = F.relu(self.bnf2(self.fc2(x)))
        out=self.fc3(x)
        return out

def get_graph_feature(x, k, idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    # print(idx.shape)
    # print(idx_base.shape)
    if idx.shape[0]>idx_base.shape[0]:
        idx=idx[idx_base.shape[0]]
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class DGCNN(nn.Module): #number of parameters: 1158722
    # from DGCNN's repo
    def __init__(self, input_channel=3,input_len=1516,features_len=64,num_classses=3,k=5,idx=None):
        super(DGCNN, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(features_len)

        self.conv1 = nn.Sequential(nn.Conv2d(input_channel*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64 * 2, features_len, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.ReLU())
        self.linear1 = nn.Linear(features_len*input_len, 128, bias=False)
        self.bn6 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout()
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout()
        self.linear3 = nn.Linear(128, num_classses)
        self.idx=idx
        if self.idx is None:
            print('original idx')
        else:
            print('redefined idx')


    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k,idx=self.idx)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k,idx=self.idx)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2), dim=1)

        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x3 = torch.cat((x1, x2), 1)
        x = F.relu(self.bn6(self.linear1(x)))
        #x = F.relu(self.bn7(self.linear2(x)))
        #x = self.dp2(x)
        out = self.linear3(x)
        return out

# class DGCNN(nn.Module): #number of parameters: 1158722
#     # from DGCNN's repo
#     def __init__(self, input_channel=3,features_len=1024,num_classses=3,k=5,idx=None):
#         super(DGCNN, self).__init__()
#         self.k = k
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm1d(features_len)
#
#         self.conv1 = nn.Sequential(nn.Conv2d(input_channel*2, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2),
#                                    nn.Dropout())
#         self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2),
#                                    nn.Dropout())
#         self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2),
#                                    nn.Dropout())
#         self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2),
#                                    nn.Dropout())
#         self.conv5 = nn.Sequential(nn.Conv1d(512, features_len, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         # self.linear1 = nn.Linear(features_len * 2, 256, bias=bias)
#         # self.linear2 = nn.Linear(256, 64, bias=bias)
#         # self.linear3 = nn.Linear(64, num_classses, bias=bias)
#         self.linear1 = nn.Linear(features_len*2, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout()
#         self.linear2 = nn.Linear(512, 256)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.dp2 = nn.Dropout()
#         self.linear3 = nn.Linear(256, num_classses)
#         self.idx=idx
#         if self.idx is None:
#             print('original idx')
#         else:
#             print('redefined idx')
#
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = get_graph_feature(x, k=self.k,idx=self.idx)
#         x = self.conv1(x)
#         x1 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x1, k=self.k,idx=self.idx)
#         x = self.conv2(x)
#         x2 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x2, k=self.k,idx=self.idx)
#         x = self.conv3(x)
#         x3 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x3, k=self.k,idx=self.idx)
#         x = self.conv4(x)
#         x4 = x.max(dim=-1, keepdim=False)[0]
#
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#
#         x = self.conv5(x)
#         x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
#         x = torch.cat((x1, x2), 1)
#         # x=self.linear1(x)
#         # x = self.linear2(x)
#         # out = self.linear3(x)
#         x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
#         x = self.dp1(x)
#         x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
#         x = self.dp2(x)
#         out = self.linear3(x)
#         return out

# class DGCNN(nn.Module): #number of parameters: 1158722
#     # from DGCNN's repo
#     def __init__(self, input_channel=3,features_len=1024,num_classses=3,bias=True,k=5,idx=None):
#         super(DGCNN, self).__init__()
#         self.k = k
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm1d(features_len)
#
#         self.conv1 = nn.Sequential(nn.Conv2d(input_channel*2, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv1d(512, features_len, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.linear1 = nn.Linear(features_len * 2, 256, bias=bias)
#         self.linear2 = nn.Linear(256, 64, bias=bias)
#         self.linear3 = nn.Linear(64, num_classses, bias=bias)
#         self.idx=idx
#         if self.idx is None:
#             print('original idx')
#         else:
#             print('redefined idx')
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = get_graph_feature(x, k=self.k,idx=self.idx)
#         x = self.conv1(x)
#         x1 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x1, k=self.k,idx=self.idx)
#         x = self.conv2(x)
#         x2 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x2, k=self.k,idx=self.idx)
#         x = self.conv3(x)
#         x3 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x3, k=self.k,idx=self.idx)
#         x = self.conv4(x)
#         x4 = x.max(dim=-1, keepdim=False)[0]
#
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#
#         x = self.conv5(x)
#         x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
#         x = torch.cat((x1, x2), 1)
#         x=self.linear1(x)
#         x = self.linear2(x)
#         out = self.linear3(x)
#         return out
class DGCNNs(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,features_len=1024,num_classses=3,bias=True,k=5,idx=None):
        super(DGCNNs, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(features_len)

        self.conv1 = nn.Sequential(nn.Conv2d(input_channel*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
        #                            self.bn2,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
        #                            self.bn3,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(128, features_len, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(features_len * 2, 256, bias=bias)
        #self.linear2 = nn.Linear(256, 64, bias=bias)
        self.linear3 = nn.Linear(256, num_classses, bias=bias)
        self.idx=idx
        if self.idx is None:
            print('original idx')
        else:
            print('redefined idx')


    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k,idx=self.idx)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = get_graph_feature(x1, k=self.k,idx=self.idx)
        # x = self.conv2(x)
        # x2 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = get_graph_feature(x2, k=self.k,idx=self.idx)
        # x = self.conv3(x)
        # x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k,idx=self.idx)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x=self.linear1(x)
        #x = self.linear2(x)
        out = self.linear3(x)
        return out


# def knn(x, k):
#     inner = -2 * torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx
# def get_graph_feature(x, k=20, idx=None):
#     batch_size = x.size(0)
#     num_points = x.size(2)
#     x = x.view(batch_size, -1, num_points)
#     if idx is None:
#         idx = knn(x, k=k)  # (batch_size, num_points, k)
#     device = torch.device('cuda')
#
#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
#
#     idx = idx + idx_base
#
#     idx = idx.view(-1)
#
#     _, num_dims, _ = x.size()
#
#     x = x.transpose(2,
#                     1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     feature = x.view(batch_size * num_points, -1)[idx, :]
#     feature = feature.view(batch_size, num_points, k, num_dims)
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
#     feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
#     return feature
#
#
# class DGCNN(nn.Module):
#     def __init__(self, k=20, num_classses=40):
#         super(DGCNN, self).__init__()
#         self.k = k
#
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm1d(1024)
#
#         self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout()
#         self.linear2 = nn.Linear(512, 256)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.dp2 = nn.Dropout()
#         self.linear3 = nn.Linear(256, num_classses)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = get_graph_feature(x, k=self.k)
#         x = self.conv1(x)
#         x1 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x1, k=self.k)
#         x = self.conv2(x)
#         x2 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x2, k=self.k)
#         x = self.conv3(x)
#         x3 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x3, k=self.k)
#         x = self.conv4(x)
#         x4 = x.max(dim=-1, keepdim=False)[0]
#
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#
#         x = self.conv5(x)
#         x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
#         x = torch.cat((x1, x2), 1)
#
#         x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
#         x = self.dp1(x)
#         x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
#         x = self.dp2(x)
#         x = self.linear3(x)
#         return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels,num_classses,edge_index,edge_weight,dropout=False,device='cpu'):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.bn1=nn.BatchNorm1d(64)
        self.bn2=nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.linear1 = nn.Linear(64*1516, 128)
        self.bnl1 = nn.BatchNorm1d(128)
        # self.linear2 = nn.Linear(512, 256)
        # self.bnl2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(128,num_classses)
        self.training=dropout
        self.edge_index=edge_index
        self.edge_weight = edge_weight
        self.device=device

    def forward(self, x):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        edge_index=self.edge_index
        edge_weight=self.edge_weight
        input=x
        x=x.permute(0,2,1)
        x=torch.reshape(x, (-1, input.shape[1]))
        batch=torch.reshape(torch.arange(0,input.shape[0]).repeat(input.shape[2],1).t(),(-1,)).to(self.device)
        x = F.relu(self.bn1(self.conv1(x, edge_index,edge_weight)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index,edge_weight)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index,edge_weight)))
        #x = F.dropout(x, training=self.training)
        x = torch.reshape(x, (-1,input.shape[2],x.shape[1]))
        x=x.view(x.size(0),-1)
        #x = global_max_pool(x, batch, input.shape[0])
        x = F.relu(self.bnl1(self.linear1(x)))
        #x = F.dropout(x,training=self.training)
        #x = F.relu(self.bnl2(self.linear2(x)))
        x = self.linear3(x)
        return x

class GCN1(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel=3,num_classes=2,A=None,features_len=64,bias=True):
        super(GCN1, self).__init__()
        self.conv1 = PYGCNConv(input_channel, 64)
        self.conv2 = PYGCNConv(64, 64)
        self.conv3 = PYGCNConv(64, features_len)
        self.bn1=nn.BatchNorm1d(64)
        self.bn2=nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(features_len)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(features_len*1516, 128),
            nn.BatchNorm1d(128, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )
        self.A=A


    def forward(self, x):
        #A = self.A
        id=numpy.random.randint(0,4)
        A=self.A[id]
        #A = torch.from_numpy(A).to(self.device)#x:Batch_size*Channels*Num_points
        x=torch.transpose(x,1,2)
        #print(x.shape)
        batch_size=x.shape[0]
        n_points=x.shape[1]
        x=torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2]))
        x = F.dropout(F.relu(self.bn1(self.conv1(x, A))))
        x =  F.dropout(F.relu(self.bn2(self.conv2(x, A))))
        x = F.relu(self.bn3(self.conv3(x, A)))

        x = torch.reshape(x, (batch_size , n_points, x.shape[-1]))
        x = x.view(x.size(0), -1)
        #print(x.size())
        out = self.classifier(x)
        return out

# class GCN1(nn.Module):
#     # from DGCNN's repo
#     def __init__(self, input_channel=3,num_classes=2,A=None,features_len=128,bias=True):
#         super(GCN1, self).__init__()
#         self.conv1 = PYGCNConv(input_channel, 64)
#         self.conv2 = PYGCNConv(64, 64)
#         self.conv3 = PYGCNConv(64, 64)
#         self.conv4 = PYGCNConv(64, 128)
#         self.conv5 = PYGCNConv(128, features_len)
#         self.bn1=nn.BatchNorm1d(64)
#         self.bn2=nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(64)
#         self.bn4 = nn.BatchNorm1d(128)
#         self.bn5 = nn.BatchNorm1d(features_len)
#         self.linear1 = nn.Linear(features_len*1516, 128, bias=bias)
#         self.bnl1 = nn.BatchNorm1d(128)
#         self.linear2 = nn.Linear(128, num_classes, bias=bias)
#         self.A=A
#
#
#     def forward(self, x):   #x:Batch_size*Channels*Num_points
#         x=torch.transpose(x,1,2)
#         #print(x.shape)
#         batch_size=x.shape[0]
#         n_points=x.shape[1]
#         A= self.A
#         x=torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2]))
#         x = F.relu(self.bn1(self.conv1(x, A)))
#         x = F.relu(self.bn2(self.conv2(x, A)))
#         x = F.relu(self.bn3(self.conv3(x, A)))
#         x = F.relu(self.bn4(self.conv4(x, A)))
#         x = F.relu(self.bn5(self.conv5(x, A)))
#         x = torch.reshape(x, (batch_size , n_points, x.shape[-1]))
#         x = x.max(dim=-2, keepdim=False)[0]
#         x = x.view(x.size(0), -1)
#         #print(x.size())
#         x = F.relu(self.bnl1(self.linear1(x)))
#         out = self.linear2(x)
#         return out

class MGR(nn.Module):
    def __init__(self, C_in, C_out, dim_k=32, heads=8):
        super().__init__()
        self.heads = heads
        self.k = dim_k

        assert (C_out % heads) == 0, 'values dimension must be integer'
        dim_v = C_out // heads

        self.conv_q = nn.Conv1d(C_in, dim_k * heads, 1, bias=False)
        self.conv_k = nn.Conv1d(C_in, dim_k, 1, bias=False)
        self.conv_v = nn.Conv1d(C_in, dim_v, 1, bias=False)

        self.norm_q = nn.BatchNorm1d(dim_k * heads)
        self.norm_v = nn.BatchNorm1d(dim_v)

        self.blocker = nn.BatchNorm1d(C_out)
        self.skip = nn.Conv1d(C_out, C_out, 1)

        # multi-dimensional adjacency matrix
        self.A = nn.Parameter(torch.randn(dim_v, dim_v, dim_k), requires_grad=True)

    def forward(self, x):
        '''
        :param x: [B, C_in, N]
        :return: out: [B, C_out, N]
        '''
        query = self.conv_q(x)  # [B, head * C_k, N]
        key = self.conv_k(x)  # [B, C_k, N]
        value = self.conv_v(x)  # [B, C_v, N]

        # normalization
        query = self.norm_q(query)
        value = self.norm_v(value)
        key = key.softmax(dim=-1)

        query = rearrange(query, 'b (h k) n -> b h k n', h=self.heads)  # [B, head, C_k, N]
        k_v_attn = einsum('b k n, b v n -> b k v', key, value)  # [B, C_k, C_v]
        Yc = einsum('b h k n, b k v -> b n h v', query, k_v_attn)  # [B, N, head, C_v]
        G = einsum('b v n, w v k -> b n k w', value, self.A).contiguous()  # A*x: [B, N, C_k, C_v]
        value = rearrange(value, 'b v n -> b n (1) v').contiguous()
        G = F.relu(G + value)  # [B, N, C_k, C_v]
        Yp = einsum('b h k n, b n k v -> b n h v', query, G)  # [B, N, head, C_v]

        out = Yc + Yp
        out = rearrange(out, 'b n h v -> b n (h v)')
        out = rearrange(out, 'b n c -> b c n')
        out = self.blocker(self.skip(out))

        return F.relu(out + x)
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
def knn_point(K, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, K]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, K, dim=-1, largest=False, sorted=False)
    return group_idx
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
class transformer(nn.Module):
    def __init__(self, C_in, C_out, n_samples=None, K=20, dim_k=32, heads=8, ch_raise=64):
        super().__init__()
        self.d = dim_k
        assert (C_out % heads) == 0, 'values dimension must be integer'
        dim_v = C_out // heads

        self.n_samples = n_samples
        self.K = K
        self.heads = heads

        C_in = C_in * 2 + dim_v
        self.mlp = nn.Sequential(
            nn.Conv2d(C_in, ch_raise, 1, bias=False),
            nn.BatchNorm2d(ch_raise),
            nn.ReLU(True),
            nn.Conv2d(ch_raise, ch_raise, 1, bias=False),
            nn.BatchNorm2d(ch_raise),
            nn.ReLU(True))

        self.mlp_v = nn.Conv1d(C_in, dim_v, 1, bias=False)
        self.mlp_k = nn.Conv1d(C_in, dim_k, 1, bias=False)
        self.mlp_q = nn.Conv1d(ch_raise, heads * dim_k, 1, bias=False)
        self.mlp_h = nn.Conv2d(3, dim_v, 1, bias=False)

        self.bn_value = nn.BatchNorm1d(dim_v)
        self.bn_query = nn.BatchNorm1d(heads * dim_k)

    def forward(self, xyz, feature):
        bs = xyz.shape[0]
        n_samples=xyz.shape[1]
        # fps_idx = pt.furthest_point_sample(xyz.contiguous(), self.n_samples).long()  # [B, S]
        # new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]
        # new_feature = index_points(feature, fps_idx).transpose(2, 1).contiguous()  # [B, C, S]
        new_xyz=xyz
        new_feature=feature.transpose(2, 1).contiguous()

        knn_idx = knn_point(self.K, xyz, new_xyz)  # [B, S, K]
        neighbor_xyz = index_points(xyz, knn_idx)  # [B, S, K, 3]
        grouped_features = index_points(feature, knn_idx)  # [B, S, K, C]
        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()  # [B, C, S, K]
        grouped_points_norm = grouped_features - new_feature.unsqueeze(-1).contiguous()  # [B, C, S, K]
        # relative spatial coordinates
        relative_pos = neighbor_xyz - new_xyz.unsqueeze(-2).repeat(1, 1, self.K, 1)  # [B, S, K, 3]
        relative_pos = relative_pos.permute(0, 3, 1, 2).contiguous()  # [B, 3, S, K]

        pos_encoder = self.mlp_h(relative_pos)
        feature = torch.cat([grouped_points_norm,
                             new_feature.unsqueeze(-1).repeat(1, 1, 1, self.K),
                             pos_encoder], dim=1)  # [B, 2C_in + d, S, K]

        feature_q = self.mlp(feature).max(-1)[0]  # [B, C, S]
        query = F.relu(self.bn_query(self.mlp_q(feature_q)))  # [B, head * d, S]
        query = rearrange(query, 'b (h d) n -> b h d n', b=bs, h=self.heads, d=self.d)  # [B, head, d, S]

        feature = feature.permute(0, 2, 1, 3).contiguous()  # [B, S, 2C, K]
        feature = feature.view(bs * 1516, -1, self.K)  # [B*S, 2C, K]
        value = self.bn_value(self.mlp_v(feature))  # [B*S, v, K]
        value = value.view(bs, n_samples, -1, self.K)  # [B, S, v, K]
        key = self.mlp_k(feature).softmax(dim=-1)  # [B*S, d, K]
        key = key.view(bs, n_samples, -1, self.K)  # [B, S, d, K]
        k_v_attn = einsum('b n d k, b n v k -> b d v n', key, value)  # [bs, d, v, N]
        out = einsum('b h d n, b d v n -> b h v n', query, k_v_attn.contiguous())  # [B, S, head, v]
        out = rearrange(out.contiguous(), 'b h v n -> b (h v) n')  # [B, C_out, S]

        return new_xyz, out

class PointTrans(nn.Module):
    def __init__(self, args, output_channels=2):
        super().__init__()

        args.num_K=[20,20]
        args.dim_k=32
        args.head=8
        args.dropout=0.5
        args.emb_dims=1024

        # transformer layer
        self.tf1 = transformer(3, 128, n_samples=512, K=args.num_K[0], dim_k=args.dim_k, heads=args.head, ch_raise=64)
        self.tf2 = transformer(128, 256, n_samples=128, K=args.num_K[1], dim_k=args.dim_k, heads=args.head, ch_raise=256)

        # multi-graph attention
        self.attn = MGR(256, 256, dim_k=args.dim_k, heads=args.head)

        self.conv_raise = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.ReLU(True))

        self.cls = nn.Sequential(
            nn.Linear(args.emb_dims, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, output_channels))

    def forward(self, x):
        # input x: [B, N, 3+3]
        x=x.permute(0,2,1)
        xyz = x[..., :3]
        #assert x.size()[-1] == 6
        feature = x[..., 3:]

        xyz1, feature1 = self.tf1(xyz, feature)
        feature1 = feature1.transpose(2, 1)
        _, feature2 = self.tf2(xyz1, feature1)

        feature3 = self.attn(feature2)
        out = self.conv_raise(feature3)
        out = self.cls(out.max(-1)[0])
        return out
class PointTrans1(nn.Module):
    def __init__(self, args, output_channels=2):
        super().__init__()

        args.num_K=[20,20]
        args.dim_k=32
        args.head=8
        args.dropout=0.5
        args.emb_dims=64

        # transformer layer
        self.tf1 = transformer(3, 128, n_samples=512, K=args.num_K[0], dim_k=args.dim_k, heads=args.head, ch_raise=64)
        self.tf2 = transformer(128, 256, n_samples=128, K=args.num_K[1], dim_k=args.dim_k, heads=args.head, ch_raise=256)

        # multi-graph attention
        self.attn = MGR(256, 256, dim_k=args.dim_k, heads=args.head)

        self.conv_raise = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.ReLU(True))

        self.cls = nn.Sequential(
            nn.Linear(args.emb_dims*1516, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(p=args.dropout),
            nn.Linear(128, output_channels))

    def forward(self, x):
        # input x: [B, N, 3+3]
        x=x.permute(0,2,1)
        xyz = x[..., :3]
        #assert x.size()[-1] == 6
        feature = x[..., 3:]

        xyz1, feature1 = self.tf1(xyz, feature)
        feature1 = feature1.transpose(2, 1)
        _, feature2 = self.tf2(xyz1, feature1)

        feature3 = self.attn(feature2)
        x= self.conv_raise(feature3)
        x=x.view(x.size(0), -1)
        out = self.cls(x)
        return out

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,remove_self_loops)
from torch_sparse import spspmm
from braingraphconv import MyNNConv
class Braingnn(torch.nn.Module):
    def __init__(self, indim, R, nclass, edge_index, edge_weight, batch, pos, k=8, ratio=0.9):
        '''
        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Braingnn, self).__init__()

        self.indim = indim
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 512
        self.dim4 = 256
        self.dim5 = 8
        self.k = k
        self.R = R
        self.edge_index=edge_index
        self.edge_weight=edge_weight
        self.batch=batch
        self.pos=pos

        self.n1 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim1 * self.indim))
        self.conv1 = MyNNConv(self.indim, self.dim1, self.n1, normalize=False)
        self.pool1 = TopKPooling(self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.n2 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim2 * self.dim1))
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
        self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

        #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
        self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)
        self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, nclass)


    def forward(self, x):
        edge_index=self.edge_index
        edge_attr=self.edge_weight
        batch=self.batch
        pos=self.pos
        input=x
        x=x.permute(0,2,1)
        x=torch.reshape(x, (-1, input.shape[1]))
        x = self.conv1(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)

        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        x = self.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1,x2], dim=1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x= F.dropout(x, p=0.5, training=self.training)
        out = self.fc3(x)
        return out

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight