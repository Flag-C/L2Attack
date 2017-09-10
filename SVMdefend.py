import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from mnistNet import mnistModel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class SVMDefend(object):
    def __init__(self,model):
        self.model=model
        self.clf=SVC(decision_function_shape='ovr',kernel='linear')
        self.X_train=None
        self.y_train=None
        self.use_gpu=False
        if not self.use_gpu:
            self.model.nn=self.model.nn.cpu()

    def preparedata(self):
        self.model.nn.eval()
        for data,target in self.model.train_loader:
            if self.use_gpu:
                data, target = data.cuda(), target.cuda()
            data,target=Variable(data,volatile=True), Variable(target)
            feature = self.model.feature(data)
            if self.X_train==None:
                self.X_train=feature.cpu().data.numpy()
                self.y_train=target.cpu().data.numpy()
            else:
                self.X_train=np.vstack((self.X_train,feature.cpu().data.numpy()))
                self.y_train=np.hstack((self.y_train,target.cpu().data.numpy()))

    def train_svc(self):
        self.clf.fit(self.X_train,self.y_train)

    def predict(self,X):
        self.model.nn.eval()
        if self.use_gpu:
            X=X.cuda()
        X=Variable(X,volatile=True)
        feature = self.model.feature(X)
        feature=feature.cpu().data.numpy()
        return self.clf.predict(feature)