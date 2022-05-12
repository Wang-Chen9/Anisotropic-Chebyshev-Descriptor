import os
import os.path as osp
import argparse
import numpy as np
import gc
import torch
import torch.nn.functional as F
from datasets.scape_dataloader import SCAPE
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, DataListLoader
from nn.csconv import ChebConv
import scipy.io as sio


parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_cpu', '--uc', dest='use_cpu',default=False, action='store_true',
                    help='bool value, use gpu or not')
parser.add_argument('--gpuid', '-g', default='1', type=str, metavar='N',
                    help='GPU id to run')
parser.add_argument('--learning_rate_softmax', '--lrs', default=0.001, type=float,
                    help='the learning rate')
parser.add_argument('--weight_decay_softmax', '--wds', default=1e-4, type=float,
                    help='the weight decay')
parser.add_argument('--learning_rate_hardloss', '--lrh', default=0.0005, type=float,
                    help='the learning rate')
parser.add_argument('--weight_decay_hardloss', '--wdh', default=5e-5, type=float,
                    help='the weight decay')
parser.add_argument('--epoch_softmax', '--es', default=300, type=int,metavar='N',
                    help='the number of training iterations with softmax loss')
parser.add_argument('--epoch_hardloss', '--eh', default=0, type=int,metavar='N',
                    help='the number of training iterations with hardnet loss')
parser.add_argument('--input_desc_dims', '--idd', default=48, type=int,
                    help='the number of dimensions in input descriptors')
parser.add_argument('--output_desc_dims', '--odd', default=256, type=int,
                    help='the number of dimensions in output descriptors')
parser.add_argument('--n_corr_points', default=5000, type=int,
                    help='the number of corresponding points')
parser.add_argument('--save_freq', '--sf', default=50, type=int,
                    help=r'save the current trained model every {save_freq} iterations')
parser.add_argument('--loading_name', '--ln', default=None, type=str,
                    help='the name of loaded model and the name of directory to generate descriptors using the loaded model')
parser.add_argument('--load', '-l', dest='load',default=False, action='store_true',
                    help='bool value, load variables from saved model or not')
parser.add_argument('--saving_name', '--sn', default='CSMCNN_Scape', type=str,
                    help='the name of trained models and the name of directory to save output descriptors')

args = parser.parse_args()

USE_GPU = not args.use_cpu

torch.backends.cudnn.benchmark = True
device_type = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'
device = torch.device(device_type)

LOAD = args.load # True
GEN = args.generate_desc # True
TRIPLET = False
EPOCH_softmax = args.epoch_softmax

SAVE_NAME = args.saving_name
CPOINT_NAME = args.loading_name
LEARNING_RATE=args.learning_rate_softmax
WEIGHT_DECAY=args.weight_decay_softmax

path = osp.join(osp.abspath('.'), 'datasets', 'SCAPE')
path_output = osp.join(osp.abspath('.'), 'outputs', SAVE_NAME)
if not os.path.exists(path_output):
    os.makedirs(path_output)
LOG_FOUT = open(path_output + '/log.out', 'w')

pre_transform = T.FaceToEdge()


train_dataset = SCAPE(path, True, None, pre_transform)
test_dataset = SCAPE(path, False, None, pre_transform)

train_loader =  DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
d = test_dataset[0]
d.num_nodes = args.n_corr_points


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_class = n_class

        self.fc1 = torch.nn.Linear(args.input_desc_dims, 64)
        self.bf1 = torch.nn.BatchNorm1d(64, eps=1e-3, momentum=1e-3)

        self.conv1 = ChebConv(64, 64)
        self.b1 = torch.nn.BatchNorm1d(64, eps=1e-3, momentum=1e-3)
        self.conv2 = ChebConv(64, 64)
        self.b2 = torch.nn.BatchNorm1d(64, eps=1e-3, momentum=1e-3)
        self.conv3 = ChebConv(64, 128)
        self.b3 = torch.nn.BatchNorm1d(64, eps=1e-3, momentum=1e-3)

        self.fc2 = torch.nn.Linear(128, args.output_desc_dims)
        self.bf2 = torch.nn.BatchNorm1d(args.output_desc_dims, eps=1e-3, momentum=1e-3)
        self.fc3 = torch.nn.Linear(args.output_desc_dims, d.num_nodes)

    def forward(self, data):
        x, V, D, A = data.x, data.V[:,0:64], data.D[0:64,:], data.A

        x = F.relu(self.bf1(self.fc1(x)))

        x = F.relu(self.b1(self.conv1(x, V, D, A)))
        x = F.relu(self.b2(self.conv2(x, V, D, A)))
        x = F.relu(self.b3(self.conv3(x, V, D, A)))

        desc = F.relu(self.bf2(self.fc2(x)))
        x = F.dropout(desc, training=self.training)
        x = self.fc3(x)

        return  F.log_softmax(x, dim=1),desc




model = Net().to(device)
target = torch.arange(d.num_nodes, dtype=torch.long, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY) # 0.01  softmax: lr=0.001, weight_decay=5e-4 triplet: lr=0.0001, weight_decay=5e-5  cheb lr=0.0005, weight_decay=1e-4



def train(epoch):
    model.train()

    loss_value = 0.0
    count = 0
    flag = True
    for data in train_loader:
        optimizer.zero_grad()
        if flag:
            x, des = model(data.to(device))  # , nloss
            x=x[data.y,:]
            loss=F.nll_loss(x, target) # + 1e-2*nloss
            loss.backward()
        optimizer.step()
        loss_value = loss_value + loss.item()
        count = count + 1
        gc.collect()

    print('Epoch: {:02d}, en-Loss: {:.4f}'.format(epoch, loss_value / count))
    LOG_FOUT.write('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss_value / count) + '\n')
    LOG_FOUT.flush()


if LOAD:
    model.load_state_dict(torch.load(osp.join(osp.abspath('.'), 'checkpoints', CPOINT_NAME + '.pth'), map_location=device_type))


for epoch in range(1, EPOCH_softmax+1):
    
    train(epoch)

    if epoch % args.save_freq == 0:
        torch.save(model.state_dict(), osp.join(osp.abspath('.'), 'checkpoints', SAVE_NAME + str(-epoch) + '.pth'))


file=osp.join(path,'files_test.txt')
with open(file, 'r') as f:
    names = [line.rstrip() for line in f]

model.eval()
with torch.no_grad():
    for idx, data in enumerate(test_dataset):
        mat_path=osp.join(path_output,names[idx])
        preds, desc = model(data.to(device))
        sio.savemat(mat_path.format(i), {'pred': pred.to('cpu').numpy().astype(np.int64),
                                         'desc': desc.to('cpu').numpy().astype(np.float32)})

LOG_FOUT.close()
