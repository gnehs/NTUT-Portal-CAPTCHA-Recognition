import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchviz import make_dot
from utils import load_data, DEVICE, human_time
from timeit import default_timer as timer


class Net(nn.Module):
    def __init__(self, gpu=False):
        super(Net, self).__init__()
        # size: 3 * 39 * 135
        self.conv1 = nn.Conv2d(3, 18, 8)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(18, 48, 8)
        self.pool2 = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        x = torch.randn(1, 3, 39, 135)
        x = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(x))))))
        self.features_size = torch.prod(torch.tensor(x.size()[1:]))
        self.fc1 = nn.Linear(self.features_size, 360)
        self.fc2 = nn.Linear(360, 4 * 26) 
        
        if gpu:
            self.to(DEVICE)
            if str(DEVICE) == 'cpu':
                self.device = 'cpu'
            else:
                self.device = torch.cuda.get_device_name(0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.features_size)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 4, 26)
        return x

    def save(self, name, folder='./models'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, name)
        torch.save(self.state_dict(), path)

    def load(self, name, folder='./models'):
        path = os.path.join(folder, name)
        map_location = 'cpu' if self.device == 'cpu' else 'gpu'
        static_dict = torch.load(path, map_location)
        self.load_state_dict(static_dict)
        self.eval()

    def graph(self):
        x = torch.rand(1, 3, 39, 135)
        y = self(x)
        return make_dot(y, params=dict(self.named_parameters()))


def loss_batch(model, loss_func, data, opt=None):
    xb, yb = data['image'], data['label']
    batch_size = len(xb)

    # Forward pass
    out = model(xb)

    # Need to reshape yb to be the same as out if not the same
    if yb.shape != out.shape:
        yb = yb.view(out.shape)

    # Compute loss
    loss = loss_func(out, yb)

    single_correct, whole_correct = 0, 0
    if opt is not None:
        # Zero grad, backward pass and weight update
        opt.zero_grad()
        loss.backward()
        opt.step()
    else:
        # calculate accuracy
        _, ans = torch.max(yb, dim=2)
        _, predicted = torch.max(out, dim=2)
        compare = (predicted == ans)
        single_correct = compare.sum().item()
        for i in range(batch_size):
            if compare[i].sum().item() == 4:
                whole_correct += 1

    loss_item = loss.item()
    del out
    del loss
    return loss_item, single_correct, whole_correct, batch_size

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, verbose=None):
    max_acc = 0
    patience_limit = 2
    patience = 0
    for epoch in range(epochs):
        patience += 1
        running_loss = 0.0
        total_nums = 0
        model.train()  # train mode
        for i, data in enumerate(train_dl):
            loss, _, _, s = loss_batch(model, loss_func, data, opt)
            if isinstance(verbose, int):
                running_loss += loss * s
                total_nums += s
                if i % verbose == verbose - 1:
                    ave_loss = running_loss / total_nums
                    print('[Epoch {}][Batch {}] got training loss: {:.6f}'
                          .format(epoch + 1, i + 1, ave_loss))
                    total_nums = 0
                    running_loss = 0.0

        model.eval()  # validate mode, working for drop out layer.
        with torch.no_grad():
            losses, single, whole, batch_size = zip(
                *[loss_batch(model, loss_func, data) for data in valid_dl]
            )
        total_size = np.sum(batch_size)
        val_loss = np.sum(np.multiply(losses, batch_size)) / total_size
        single_rate = 100 * np.sum(single) / (total_size * 4)
        whole_rate = 100 * np.sum(whole) / total_size
        if single_rate > max_acc:
            patience = 0
            max_acc = single_rate
            model.save('pretrained')

        print('After epoch {}: \n'
              '\tLoss: {:.6f}\n'
              '\tSingle Acc: {:.2f}%\n'
              '\tWhole Acc: {:.2f}%'
              .format(epoch + 1, val_loss, single_rate, whole_rate))
        if patience > patience_limit:
            print('Early stop at epoch {}'.format(epoch + 1))
            break


def train(use_gpu=True):
    train_dl, valid_dl = load_data(batch_size=4, split_rate=0.2, gpu=use_gpu)
    model = Net(use_gpu)
    opt = optim.Adadelta(model.parameters())
    criterion = nn.BCEWithLogitsLoss()   # loss function
    start = timer()
    fit(50, model, criterion, opt, train_dl, valid_dl, 500)
    end = timer()
    t = human_time(start, end)
    print('Total training time using {}: {}'.format(model.device, t))


if __name__ == '__main__':
    train(True)
