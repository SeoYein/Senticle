#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 23:45:19 2018

@author: pirl


"""

import os
import time
import datetime
import csv
from tensorflow import flags
import tensorflow as tf
import numpy as np
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer
import cnn_tool as tool
import pandas as pd

os.chdir('/home/pirl/Downloads/CNN')
# making preprocessed file with soynlp
if os.path.isfile('pre_posco'+'.csv')==False:
    print("\n")
    print('pre_posco'+'.csv" is not EXISTS!')
    print("\n")
    doc = pd.read_csv('posco_4Y_data.csv')
    contents = []
    drop_list = []
    label =[]
    date = []
    for i in range(len(doc['text'])):
        if len(doc.iloc[i]['text']) > 100:
            date.append(doc.iloc[i]['datetime'])
            contents.append(doc.iloc[i]['text']) # contents
            label.append(doc.iloc[i]['num'])
        else:
            drop_list.append(i)
    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(contents, min_noun_frequency=50)
    
    
    match_tokenizer = NounLMatchTokenizer(nouns)
    
    f = open('pre_posco' + '.csv', 'w', newline='', encoding='utf-8')
    fieldnames = ['datetime','text', 'label']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    test = []
    for j in range(len(contents)):
        temp_list = match_tokenizer.tokenize(contents[j])
        del_list2 = []
        for i in range(len(temp_list)):
            if len(temp_list[i])==1: #자른 워드 크기 1이면 삭제
                del_list2.append(i)
            elif type(temp_list[i])==float:
                del_list2.append(i)
            else:
                pass
        del_list2.sort(reverse = True)
        for i in del_list2:
            try:
                del temp_list[i]
            except ValueError:
                pass
        temp_list = ' '.join(temp_list)
        len(temp_list)
        test.append(temp_list)
        writer.writerow({'datetime':date[j],'text': temp_list, 'label' : label[j] })
        if j % 10 == 0:
            print("{}개의 기사 중 {}번 기사 불용어처리후 저장완료~ ^오^".format(len(contents), j+1))
    f.close()
    contents = test.copy()
    print('기사 갯수 : ',len(contents))
    print("사전 생성 완료 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    
#%% 유사도 점검을 통한 중복기사 제거
import os 
os.chdir('/home/pirl/Downloads/CNN')
pre_df = pd.read_csv('pre_posco.csv', )
df = pd.read_csv('posco_4Y_data.csv')
df['text']


# delete article lower then 100 words
''' Done Before
drop_list=[]
for i in range(len(df['text'])):
        if len(df.iloc[i]['text']) > 100:
            pass
        else:
            drop_list.append(i)
            
df.drop(df.index[drop_list], inplace=True)
'''

# check : length of df and pre_df should be same
#len(df)
#df.text.dropna(axis=0)
len(pre_df)
pre_df.dropna(axis=0)




from sklearn.feature_extraction.text import TfidfVectorizer

mydoclist_soy = []
for i in range(len(pre_df.text)):
    mydoclist_soy.append(pre_df.text[i])
    
# fix error about NA   
check = pd.DataFrame(mydoclist_soy)
check2 = check.dropna(axis=0)   
na_list=[]
for i in check.index.tolist():
    if i not in check2.index.tolist():
        na_list.append(i)

for i in na_list:
    del mydoclist_soy[i]
len(mydoclist_soy)    
pre_df.drop(pre_df.index[na_list], inplace=True)

tfidf_vectorizer = TfidfVectorizer(min_df = 1)
tfidf_matrix_soy = tfidf_vectorizer.fit_transform(mydoclist_soy)


# doc distance
document_distances_soy = (tfidf_matrix_soy * tfidf_matrix_soy.T)
print('\n')
print('soy를 활용한 유사도 분석을 위해 '+str(document_distances_soy.get_shape()[0])+'x'+str(document_distances_soy.get_shape()[1])+'matrix를 만들었따뤼~ ^오^')
print('\n')
print(document_distances_soy.toarray())
ar = document_distances_soy.toarray()

# for easy looking
df_features = pd.DataFrame(ar, index=pre_df.index.tolist())


# simplicity
ar_lst = []
for i in range(len(pre_df)):
    print('전체 %d개 인덱스 중 %d번째 인덱스 체크 중...'%(len(pre_df),i))
    for j in range(i+1,len(ar)):
        if ar[i][1+j-1]>0.35:
            if pre_df.label.iloc[j] != pre_df.label.iloc[i]: # about label
                ar_lst.append(j)
            
            
# remove duplication
ar_lst = list(set(ar_lst))
pre_df.iloc[ar_lst]

pre_df.drop(pre_df.index[ar_lst], inplace=True)


# checking right labeling
lcheck = pd.merge(pre_df, df, on ='datetime') # inner merge
lcheck = lcheck.dropna(axis=0) # because of key value(datetime), len(lcheck) > len(pre_df)

a=[]
for i in range(len(lcheck)):
    if lcheck['label'][i] + lcheck['num'][i]==1:
        a.append(lcheck.loc[i])

len(a) # should be 0 

lcheck.to_csv('lcheck.csv', index = False)

# file extraction
#df.to_csv('repro_posco_4Y_55.csv', sep=',', na_rep='NaN', index = False)
pre_df.to_csv('repro_35.csv',index = False)


len(pre_df)
