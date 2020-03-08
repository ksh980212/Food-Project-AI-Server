#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, jsonify
from torch import FloatTensor
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext.data import TabularDataset
import numpy as np
import pandas as pd
import random

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(705, 200),
            nn.Linear(200, 100),        
            nn.ReLU(True), #이건 왜쓴거지? 모르겠음
            nn.Linear(100, 1))
        self.decoder = nn.Sequential(
            nn.Linear(1, 100),
            nn.Linear(100, 200),        
            nn.ReLU(True),
            nn.Linear(200, 705), nn.Tanh())
            # nn.Linear(200, 702), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def autoencoder_train():
    num_epochs = 40
    batch_size = 100

    learning_rate = 1e-3

    #t1_data = pandas dataframe
    t1_data = pd.read_csv('./data/user_data.csv')

    #사용자수, 음식 종류의 수
    nb_users = int(max(t1_data.iloc[:,0])) + 1 #사용자의 수 +1
    nb_foods = int(max(t1_data.iloc[:,1])) + 1#음식 종류의 수 만약 인덱스 0부터 주었다면 +1

    #pandas dataframe -> numpy
    t1_data = t1_data.values

    #numpy array -> pytorch tensor
    def convert(data):
        new_data = []
        for id_users in range(0, nb_users): #총 사용자 수많큼 반복해라
            id_foods = data[:,1][data[:,0] == id_users] # user가 본 영화들
            id_foods = id_foods.astype(int) # user가 본 영화들의 별점
            id_ratings = data[:,2][data[:,0] == id_users]
            ratings = np.zeros(nb_foods) #영화 숫자만큼 zero 배열 만들어줌
            ratings[id_foods] = id_ratings #id_movies영화갯수 1부터 하려고 -1을 해줌/ id_movies - 1번째 영화 /ratings[id_movies - 1]: n번째 영화 별점이 몇점인지 쭉 나열
            ratings = ratings.astype(float)
            new_data.append(list(ratings)) #전체영화 zero에 배열되있는것에 점수 넣어줌
        return new_data

    t2_data = convert(t1_data)
    t2_data = np.asarray(t2_data)

    # Numpy array to Pytorch Tensor
    tensor = torch.FloatTensor(t2_data)
    num_train_dataset = int(len(tensor) * 0.8)
    num_test_dataset = len(tensor) - num_train_dataset

    train_dataset, test_dataset = torch.utils.data.random_split(tensor, [num_train_dataset, num_test_dataset])

    dataloader = DataLoader(tensor, batch_size=batch_size, shuffle=True)


    model = autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    for epoch in range(num_epochs):
        for data in dataloader:
            output = model(data)
            loss = criterion(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch +1, num_epochs, loss.item()))
    
    return model


