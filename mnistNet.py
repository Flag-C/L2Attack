import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
CUDA=True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class mnistModel(object):
    def __init__(self):
        self.traindata=datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ]))
        self.testdata=datasets.MNIST('../data', train=False,
                                     transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
            batch_size=1000, shuffle=True, )
        self.nn = Net()
        if CUDA:
            self.nn.cuda()

        self.optimizer = optim.SGD(self.nn.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)

    def extract(self, x):
        x = F.relu(F.max_pool2d(self.nn.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.nn.conv2_drop(self.nn.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.nn.fc1(x))
        x = self.nn.fc2(x)
        return x

    def feature(self,x):
        x = F.relu(F.max_pool2d(self.nn.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.nn.conv2_drop(self.nn.conv2(x)), 2))
        x = x.view(-1, 320)
        return x

    def train(self,epoch):
        self.nn.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if CUDA:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.nn(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data[0]))

    def test(self):
        self.nn.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if CUDA:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.nn(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def getModel(self,epochs=5):
        for epoch in range(epochs):
            self.train(epoch)
            self.test()

    def pred(self,data):
        self.nn.eval()
        output=self.nn(data)
        pred=output.data.max(1,keepdim=True)[1]
        return pred