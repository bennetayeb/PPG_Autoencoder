import codecs
import argparse
import os

import matplotlib.pyplot as plt
import tqdm
import numpy
import glob
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
import torch
import torch.nn as nn
import torch.nn.utils.rnn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
import codecs
from torchsummary import summary
from torchshape import tensorshape
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

hyper = {
    "randomSeed": 42,
    "nEpochs":70,
    "PATH": "model.pt",
    "batchSize":4,
    "lr":1e-3,
    "clip_grad":0.1,

}
torch.manual_seed(hyper["randomSeed"]);
device = torch.device("cpu")


filename = 'data.txt'
dir_fpath = '/home/rania/Documents/workspace/IntraSpkVC/data_fr/ppg/FFR0009'
file_list = glob.glob(dir_fpath + '/*.npy')
# print(len(file_list))
data = []
vectorLengths = []
for idx, s in enumerate(file_list):
    vector = numpy.load(s)
    data.append(torch.tensor(vector))
    data.sort(key=len)
    vectorLengths.append(len(vector))
    vectorLengths.sort()

class Dataset(Dataset):
    def __init__(self, data):
        # self.vectorLengths = vectorLengths
        # self.data = data
        self.data = pad_sequence(data, batch_first=True, padding_value=0)

    def __getitem__(self, idx):
        x = self.data[idx].clone().detach().requires_grad_(True)
        x = torch.unsqueeze(x, 0)
        return x

class CustomDataset(Dataset):
    def __init__(self, data):
        # self.vectorLengths = vectorLengths
        self.data = data

    def __len__(self):
        #or use input_lengths
        # print(self.vactorLengths)
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone().detach().requires_grad_(True)
        # x = torch.unsqueeze(x, 0)
        return x

dataPadded = Dataset(data=data)

trainDataX, testDataX,trainDataY, testDataY = train_test_split([data for data in dataPadded ],[data for data in dataPadded ],test_size=0.2,random_state=41,shuffle=False,stratify=None)
# print("size of the training dataset {}".format(len(trainDataX)), trainDataX[2].shape)
# print("size of the training dataset {}".format(len(trainDataX)), trainDataX[3].shape)
valDataX, testDataX, valDataY, testDataY = train_test_split(testDataX,testDataY,test_size=0.5,random_state=41,shuffle=False,stratify=None)
# print("size of the validation dataset {}".format(len(valDataX)), valDataY[0].shape)
# print("size of the validation dataset {}".format(len(valDataX)), valDataY[1].shape)

def masking(data):
    pad = 0
    dataMask = (data == pad).type(torch.int16)
    dataMask = 1 - dataMask
    maskedData = torch.mul(dataMask, data)
    return maskedData

def train(trainLoader, model, criterion, optimizer,epoch =hyper["nEpochs"]):
    model.train()
    totalLoss, batchNum = 0, 0
    for i, trainData in enumerate(trainLoader):
        trainData = Variable(trainData).to(device)
        # ===================forward=====================
        output = model(trainData)
        loss = criterion(masking(output), trainData)
        # print(loss)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        totalLoss += loss.item()
        batchNum += 1
    trainTotalCount = totalLoss / batchNum
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epoch, trainTotalCount))
    return trainTotalCount

def valid(valLoader, model, criterion, epoch=hyper["nEpochs"]):
    model.eval()
    with torch.no_grad():
        totalLoss, batchNum = 0, 0
        for i, valData in enumerate(valLoader):
            valData = Variable(valData).to(device)
            # ===================forward=====================
            output = model(valData)
            loss = criterion(masking(output), valData)
            totalLoss += loss.item()
            batchNum += 1
        ValTotalCount = totalLoss / batchNum
    print('epoch [{}/{}], Validation loss:{:.4f}'.format(epoch + 1, epoch, ValTotalCount))
    return ValTotalCount


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, bestValidLoss = float('inf')
    ):
        self.bestValidLoss = bestValidLoss

    def __call__(
            self, currentValidLoss,
            epoch, model, optimizer, criterion
    ):
        if currentValidLoss < self.bestValidLoss:
            self.bestValidLoss = currentValidLoss
            print(f"\nBest validation loss: {self.bestValidLoss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': hyper["nEpochs"] + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, 'outputs/best_model.pth')
saveBestModel = SaveBestModel()

def save_model(model, optimizer, criterion, epochs=hyper["nEpochs"]):
    """
    Function to save the trained model
    """
    print(f"Saving final model...")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'outputs/final_model.pth')


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # b, 16, 1768, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # b,  32, 884, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  # b, 64, 878, 1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # b, 32, 884, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,  output_padding=1),  # b, 16, 1768, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # b, 1, 3536, 28
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def main():
    datasetTrain = CustomDataset(data=trainDataX)
    trainLoader = DataLoader(datasetTrain,
                             batch_size=hyper["batchSize"],
                             shuffle=True
                             )
    # for i, data in enumerate(trainLoader):
    #     print('train', data.shape)
    datasetVal = CustomDataset(data=valDataX)

    valLoader = DataLoader(datasetVal,
                           batch_size=hyper["batchSize"],
                           shuffle=True
                           )
    # for i, data in enumerate(valLoader):
    #     print('val ', data.shape)
    model = Autoencoder().to(device)
    # model = summary(model, (1, 3536, 28))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=hyper["lr"],
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    for epoch in range(hyper["nEpochs"]):
        print(f"[INFO]: Epoch {epoch} of {hyper['nEpochs']}")
        # # ===================training========================
        trainEpochLoss = train(trainLoader, model, criterion, optimizer, epoch)
        # # ===================validation========================
        validEpochLoss = valid(valLoader, model, criterion, epoch)
        # # ===================checkpoints========================
        trainLoss, validLoss = [], []
        # start the training
        trainLoss.append(trainEpochLoss)
        validLoss.append(validEpochLoss)

        print(f"Training loss: {trainEpochLoss:.3f}")
        print(f"Validation loss: {validEpochLoss:.3f}")
        # save the best model till now if we have the least loss in the current epoch
        saveBestModel(
            validEpochLoss, epoch, model, optimizer, criterion
        )
        # save the trained model weights for a final time
        save_model(model, optimizer, criterion, epoch)
        print('TRAINING COMPLETE')


if __name__ == '__main__':
    main()

