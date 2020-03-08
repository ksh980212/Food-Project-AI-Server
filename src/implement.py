#!/usr/bin/env python
# coding: utf-8

# In[1]:


from autoencoder_train_real import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



#가중치를준 음식들 INPUT에 넣고 맛있는 순서대로 나오게끔 하기
def train_model(model, include, exclude):
    aaa=[]
    lists = include
    arr=[1 for i in range(1, 706)]
    count=0
    for i in lists:
        if count<4:
            arr[i]=5
        elif count<8:
            arr[i]=5
        elif count<12:
            arr[i]=5
        count+=1
    arr=[1 for i in range(1, 706)]

    aaa.append(arr)
    bb=torch.FloatTensor(aaa)
    new_user_input = bb
    output = model(new_user_input)

    output = (output+1)


    # 가장 맛있는 음식 순서대로 나열하기
    sort_food_id = np.argsort(-output.detach().numpy())

    # array to list
    sort_food_id_list=sort_food_id.tolist()

    # 차원 줄이기
    food_real_list=np.ravel(sort_food_id_list, order='C').tolist()
    # print(food_real_list)
    # return

    file=pd.read_csv('./data/exception_food_label.csv')

    rm_list=set()

    for j in exclude:
        for i in range(705):
            if file[j][i]==1:
                rm_list.add(file['f_num'][i])



    #타입 리스트로 바꾸기
    rm_real_list = list(rm_list)
    # print(rm_real_list)

    #food_real_list는 전체 세트 - 제거할거 제거하기
    for i in food_real_list:
        if i in rm_list or i in include:
            food_real_list.remove(i)
            
    top_10 = food_real_list[:10]


    count1 = 2
    sampleList1 = include
    for i in sampleList1:
        if i in rm_list:
            sampleList1.remove(i)
    random_list1 = random.sample(sampleList1, count1)



    count = 3
    sampleList = top_10
    random_list2 = random.sample(sampleList, count)



    #CBF코드
    Top_1 = food_real_list[:1]
    # print("Top1"+str(Top_1))

    data = pd.read_excel('./data/11.30.xlsx')

    count_vect = CountVectorizer(min_df=0, ngram_range=(1, 1))



    ingredient_mat = count_vect.fit_transform(data['ingredients_ten'])

    ingredient_sim = cosine_similarity(ingredient_mat, ingredient_mat)

    ingredient_sim_sorted_ind = ingredient_sim.argsort()[:, ::-1]
    # print("ingredient_sim_sorted_ind"+str(ingredient_sim_sorted_ind))

#여기부터 def

    similar_indexes = ingredient_sim_sorted_ind[Top_1, :(705)]
    similar_indexes = similar_indexes.reshape(-1)
    a = data.iloc[similar_indexes]
    b = a['product'].tolist()


    for i in b:
        if i in rm_list or i in Top_1:
            b.remove(i)

    # print("b:"+str(b))
    count3 = 1
    random_list3 = random.sample(b, count3)




#마지막 관문

    final_list = random_list1 + random_list2+random_list3

    plus_one=[]
    for i in final_list:
        plus_one.append(i+1)

    return plus_one

# remove_random(include, exclude)
