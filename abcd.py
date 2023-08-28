from __future__ import print_function
import numpy
import torch.utils.data as data
import torch
import random

class Classification(data.Dataset):
    def __init__(self,vec,gt,sigma=0):
        self.vec=vec
        self.label=gt
        self.sigma=sigma

    def __getitem__(self, index: int):
        feat=self.vec[index]
        label = self.label[index]
        # if self.sigma[0]>0 and label>0:
        #     noise=numpy.random.normal(0,self.sigma,feat.shape)
        #     feat+=noise
        feat = torch.tensor(feat.T, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return feat,label

    def __len__(self) -> int:
        return len(self.vec)

class Classification_sample(data.Dataset):
    def __init__(self,vec,gt,num_classes):
        self.vec=vec
        self.label=gt
        self.num_classes=num_classes

    def __getitem__(self, index: int):
        sample_class = random.randint(0, self.num_classes - 1)
        sample_indexes=numpy.where(self.label==sample_class)[0]
        index = random.choice(sample_indexes)
        feat=self.vec[index]
        label = self.label[index]
        feat = torch.tensor(feat.T, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return feat,label

    def __len__(self) -> int:
        return int(len(self.vec)/20)

class Classification_sample1(data.Dataset):
    def __init__(self,vec,gt,num_min=370):
        self.vec=vec
        self.label=gt
        self.num_min=num_min

    def __getitem__(self, index: int):
        sample_class = index%3
        sample_indexes=numpy.where(self.label==sample_class)[0]
        #index = random.choice(sample_indexes)
        index = sample_indexes[index//3]
        feat=self.vec[index]
        label = self.label[index]
        feat = torch.tensor(feat.T, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return feat,label

    def __len__(self) -> int:
        return self.num_min*3