import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.utils as utils
from src.IrisDataset import IrisDataset

data = pd.read_csv("data/Iris.csv", index_col=False)

data.loc[data['Species'] == 'Iris-setosa', 'Species'] = 0
data.loc[data['Species'] == 'Iris-versicolor', 'Species'] = 1
data.loc[data['Species'] == 'Iris-virginica', 'Species'] = 2
data = data.apply(pd.to_numeric)
msk = np.random.rand(len(data)) < 0.8
data_array_train = data[msk].as_matrix()
xtrain = data_array_train[:, :4]
ytrain = data_array_train[:, 4]

data_array_test = data[~msk].as_matrix()
xtest = data_array_test[:, :4]
ytest = data_array_test[:, 4]

hl = 10
lr = 0.01
num_epoch = 1001


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.Linear1 = nn.Linear(4, 3)
        # self.Linear2 = nn.Linear(hl, 3)

    def forward(self, x):
        # x = F.relu(self.Linear1(x))
        x = self.Linear1(x)
        return x


def trainEpoch(train_loader, model, criterion, optimizer, epoch):
    losses = utils.AverageMeter()

    model.train()

    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()

        def closure():
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            losses.update(loss.data.cpu().numpy(), labels.size(0))
            print('loss:', loss.item())
            loss.backward()
            plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)
            return loss

        optimizer.step(closure)


def valEpoch(val_loader, model, criterion, epoch):
    losses = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.data.cpu().numpy(), labels.size(0))
            _, predicted = torch.max(outputs, 1)
            if batch_idx == 0:
                out = predicted.data.cpu().numpy()
                label = labels.cpu().numpy()
            else:
                out = np.concatenate((out, predicted.data.cpu().numpy()), axis=0)
                label = np.concatenate((label, labels.cpu().numpy()), axis=0)

        # Accuracy
        acc = np.sum(out == label) / len(out)

        # Print validation info
        print('Validation set: Average loss: {:.4f}\t'
              'Accuracy {acc}'.format(losses.avg, acc=acc*100))

        # Plot validation results
        plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
        plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc*100)

        # Return acc as the validation outcome
        return acc


def trainProcess():
    # Load model
    model = Model()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = GradientDescent.GradientDescent(model.parameters(), lr=lr)
    optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)


    best_val = float(0)

    train_ds = IrisDataset("data/train.csv")
    trainloader = DataLoader(train_ds, batch_size=200,
                             shuffle=True, num_workers=1)

    test_ds = IrisDataset("data/test.csv")
    testloader = DataLoader(test_ds, batch_size=200,
                            shuffle=False, num_workers=1)
    start = time.time()
    for epoch in range(255):
        trainEpoch(trainloader, model, criterion, optimizer, epoch)
        lossval = valEpoch(testloader, model, criterion, epoch)
        best_val = max(lossval, best_val)
        print('** Validation: %f (best) - %f (current)' % (best_val, lossval))
    print('Time used is ', time.time() - start)


if __name__ == "__main__":
    # Plots
    global plotter
    plotter = utils.VisdomLinePlotter(env_name='LBFGS')

    # Training process
    trainProcess()
