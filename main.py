import codecs
import argparse
import os
import tqdm
import numpy
import glob
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.utils.rnn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchsummary import summary
from torchshape import tensorshape
from pprint import pprint


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = pad_sequence(data, batch_first=True, padding_value=0)

    def __len__(self):
        #or use input_lengths
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone().detach().requires_grad_(True)
        return torch.unsqueeze(x, 0)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        image_size = (4, 1, 28, 28)
        self.encoder = nn.Sequential(
            # nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
           # nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            # nn.ReLU(True),
            #nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2

            # Input : b, 1, 28,28
            nn.Conv2d(1, 16, 3, stride=1, padding="same"),  # b, 16, 28, 28
            #print(tensorshape(nn.Conv2d(1, 16, 3, stride=1, padding="same"), image_size)),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 14, 14
            #print(tensorshape(nn.MaxPool2d(2, stride=2), image_size)),

        )
        self.decoder = nn.Sequential(
            # #nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            # #nn.ReLU(True),
            # nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            # nn.Tanh()
            # nn.ConvTranspose2d(8, 16, 3, stride=3, padding=4), # b, 16, 14, 14
            nn.ConvTranspose2d(16, 1, 2, stride=2), # b, 1, 28, 28
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def getData():
    filename = 'data.txt'
    dir_fpath = '/home/rania/Documents/workspace/IntraSpkVC/data_fr/ppg/FFR0009'
    file_list = glob.glob(dir_fpath + '/*.npy')[:10]
    # print(file_list)

    vectors = []
    input_lengths = []

    for i, s in enumerate(file_list):
        vector = numpy.load(s)
        input_lengths.append(len(vector))
        vectors.append(torch.tensor(vector))
    # print(numpy.shape(vectors))
    # print(numpy.shape(vectors[6]))
    # print(numpy.shape(vectors[6][0]))
    return vectors
vectors = getData()


def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    return m



def main():
    dataset = CustomDataset(data=vectors)
    dataLoader = DataLoader(dataset, batch_size=4, shuffle=True)
    # dataset= numpy.asarray(dataset)
    #print('dataset 0 shape: ', dataset[0].shape)
    #print(len(dataset))






    #print(f"length dataLoader: {len(dataLoader)}")
    # print("type dataloader:", type(dataLoader))

    num_epochs = 5
    learning_rate = 1e-3


    model = Autoencoder().to(device='cpu')
    #model = summary(model, (1, 28, 28))

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)
    for epoch in range(num_epochs):
        total_loss = 0
        batch_num=0
        pad = 0
        for i,data in enumerate(dataLoader):
            pad = 0
            print('data', data.shape)
            data = Variable(data)
            dataMask = (data == pad).type(torch.int16)
            dataMask = 1 - dataMask
            print('dataMask', dataMask.shape)

            maskedData = torch.mul(dataMask, data)
            print('maskedData', maskedData.shape)
            maskedData = Variable(maskedData)

            # # ===================forward=====================

            output = model(maskedData)
            outputMask = (output == pad).type(torch.int16)
            outputMask = 1 - outputMask
            maskedOutput = torch.mul(outputMask, output)
            loss = criterion(maskedOutput, maskedData)
            #print(loss)

            # # ===================backward====================

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            batch_num += 1

            # # ===================log========================

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, total_loss/batch_num))


        #torch.save(model.state_dict(), './conv_autoencoder.pth')


if __name__ == '__main__':
    main()
