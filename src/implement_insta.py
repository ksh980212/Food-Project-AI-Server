#!/usr/bin/env python
# coding: utf-8

# In[1]:

from autoencoder_train_real import *


#가중치를준 음식들 INPUT에 넣고 맛있는 순서대로 나오게끔 하기
def insta_model(model,data):
    aaa=[]
    lists = data
    arr=[0 for i in range(1, 706)]
    count=0
    for i in lists:
        if count<4:
            arr[i]=5
        elif count<8:
            arr[i]=3
        elif count<12:
            arr[i]=1
        count+=1


    aaa.append(arr)
    bb=torch.FloatTensor(aaa)
    new_user_input = bb
    output = model(new_user_input)

    output = (output+1)


    # 가장 맛있는 음식 순서대로 나열하기
    sort_food_id = np.argsort(-output.detach().numpy())

    final_insta=sort_food_id[:10]

    # array to list
    sort_food_id_list=sort_food_id.tolist()

    # 차원 줄이기
    food_real_list=np.ravel(sort_food_id_list, order='C').tolist()

    count1 = 6
    sampleList1 = food_real_list
    sampleList1 = random.sample(sampleList1, count1)

    final_list = sampleList1

    plus_one=[]
    for i in final_list:
        plus_one.append(i+1)


    return plus_one



