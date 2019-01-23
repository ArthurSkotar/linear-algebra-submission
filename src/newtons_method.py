import torch
from src.Newton import Model
import pandas as pd
from torch.autograd import Variable

if __name__ == '__main__':
    model = Model()
    model.load_state_dict(torch.load("model/newton.pt"))
    model.eval()
    data_test = pd.read_csv("data/train.csv", index_col=False)
    data_array_test = data_test.as_matrix()
    xtest = data_array_test[:, :4]
    ytest = data_array_test[:, 4]
    X = Variable(torch.Tensor(xtest).float())
    Y = torch.Tensor(ytest).long()
    out = model(X)
    _, predicted = torch.max(out.data, 1)

    # get accuration
    print('Accuracy of the network %d %%' % (100 * torch.sum(Y == predicted) / len(Y)))

