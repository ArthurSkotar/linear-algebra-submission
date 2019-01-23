import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.functional import F

import src.utils as utils


def eval_hessian(loss_grad, model):
    l = loss_grad.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = torch.autograd.grad(loss_grad[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return torch.inverse(hessian.cpu().data)


def get_second_order_grad(grads, xs):
    start = time.time()
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        print('2nd order on ', j, 'th layer')
        print(x.size())
        grad = torch.reshape(grad, [-1])
        grads2_tmp = []
        for count, g in enumerate(grad):
            g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp.append(g2[count].data.cpu().numpy())
        grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())))
        print('Time used is ', time.time() - start)
    for grad in grads2:  # check size
        print(grad.size())

    return grads2


def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=False,
                                retain_graph=True,
                                create_graph=True)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.Linear1 = nn.Linear(4, 3)
        # self.Linear2 = nn.Linear(hl, 3)

    def forward(self, x):
        # x = F.relu(self.Linear1(x))
        x = self.Linear1(x)
        return F.log_softmax(x, dim=1)


def validate(test_data_x, test_data_y, model, criterion, epoch):
    # get prediction
    losses = utils.AverageMeter()
    model.eval()
    X = Variable(torch.Tensor(test_data_x).float())
    Y = torch.Tensor(test_data_y).long()
    out = model(X)
    loss = criterion(out, Y)
    _, predicted = torch.max(out.data, 1)
    torch_sum = torch.sum(Y == predicted)
    print(torch_sum)
    acc = 100 * torch_sum / len(Y)
    # Print validation info
    losses.update(loss.data.cpu().numpy(), outputs.size(0))

    print('Validation set: Average loss: {:.4f}\t'
          'Accuracy {acc}'.format(losses.avg, acc=acc))

    plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)
    print('Accuracy of the network %d %%' % (acc))


if __name__ == '__main__':
    data_train = pd.read_csv("data/train.csv", index_col=False)
    data_test = pd.read_csv("data/test.csv", index_col=False)
    data_array_train = data_train.as_matrix()
    xtrain = data_array_train[:, :4]
    ytrain = data_array_train[:, 4]
    X = Variable(torch.Tensor(xtrain).float())
    Y = Variable(torch.Tensor(ytrain).long())
    data_array_test = data_test.as_matrix()
    xtest = data_array_test[:, :4]
    ytest = data_array_test[:, 4]
    model = Model()

    loss_function = nn.CrossEntropyLoss()
    global plotter
    plotter = utils.VisdomLinePlotter(env_name='newton')

    threshold = 1e-5
    a = 0.01
    j = 0
    max_iter = 250
    prev_loss = 9999
    start = time.time()
    while j < max_iter:
        losses = utils.AverageMeter()
        model.train()
        outputs = model(X)
        loss = loss_function(outputs, Y)
        if abs(prev_loss - loss) < threshold:
            break
        losses.update(loss.data.cpu().numpy(), outputs.size(0))
        print(loss)
        gn = gradient(loss, model.parameters())
        xs = torch.cat([x.contiguous().view(-1) for x in model.parameters()])
        values = eval_hessian(gn, model)
        if j == 200:
            a = 0.001

        d = values @ gn
        xs = xs - a * d
        i = 0
        j += 1
        for p in model.parameters():
            shape = p.shape
            ind = shape[0] * (shape[1] if len(shape) > 1 else 1)
            ind_ = xs[i:i + ind].view(shape)
            p.data.add_(- p.data + ind_)
            i += ind
        prev_loss = loss
        # Plot loss after all mini-batches have finished
        plotter.plot('loss', 'train', 'Class Loss', j, losses.avg)
        validate(test_data_x=xtest, test_data_y=ytest, model=model, criterion=loss_function, epoch=j)
    print('Time used is ', time.time() - start)
    torch.save(model.state_dict(), "model/newton.pt")
