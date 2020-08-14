# -*- coding: utf-8 -*-
"""
Teste: Cross Validation

@author: edvonaldo
"""

import os
import csv
import math
import random
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# Leitura
path_al2014 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/AL_2014.csv'
path_al2015 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/AL_2015.csv'
path_al2016 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/AL_2016.csv'
path_al2017 = 'C:/Users/edvon/Google Drive/UFAL/TCC/CODES/tcc_codes/read_csv_files/AL_data.csv'
path_al2018 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/AL_2018.csv'


data_al2014 = pd.read_csv(path_al2014)
data_al2015 = pd.read_csv(path_al2015)
data_al2016 = pd.read_csv(path_al2016)
data_al2017 = pd.read_csv(path_al2017)
data_al2018 = pd.read_csv(path_al2018)


#%% LIMPANDO

#% 2.1 - Limpeza

del data_al2014['Unnamed: 0']
del data_al2015['Unnamed: 0']
del data_al2016['Unnamed: 0']
del data_al2017['Unnamed: 0']
del data_al2018['Unnamed: 0']

# Escolhendo apenas as colunas de interesse
data_al2014 = data_al2014.loc[:,'NT_GER':'QE_I26']
data_al2015 = data_al2015.loc[:,'NT_GER':'QE_I26']
data_al2016 = data_al2016.loc[:,'NT_GER':'QE_I26']
data_al2017 = data_al2017.loc[:,'NT_GER':'QE_I26']
data_al2018 = data_al2018.loc[:,'NT_GER':'QE_I26']

data_al2014 = data_al2014.drop(data_al2014.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
data_al2015 = data_al2015.drop(data_al2015.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
data_al2016 = data_al2016.drop(data_al2016.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
data_al2017 = data_al2017.drop(data_al2017.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)
data_al2018 = data_al2018.drop(data_al2018.loc[:, 'CO_RS_I1':'CO_RS_I9'].columns, axis=1)

data_al2014 = data_al2014.drop(data_al2014.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
data_al2015 = data_al2015.drop(data_al2015.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
data_al2016 = data_al2016.drop(data_al2016.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
data_al2017 = data_al2017.drop(data_al2017.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)
data_al2018 = data_al2018.drop(data_al2018.loc[:, 'NT_FG':'NT_CE_D3'].columns, axis=1)

#%% MERGE NOS DADOS: data al
# Observando os dados
#print('O formato dos dados é: ', features_al.shape)

#describe_al = features_al.describe()

#print('Descrição para as colunas: ', describe_al)
#print(describe_al.columns)
frames = [data_al2014, data_al2015, data_al2016, data_al2017, data_al2018]
data_al = pd.concat(frames)

#%% AJUSTANDO

# Números que são strings para float
# Colunas NT_GER a NT_DIS_FG ^ NT_CE a NT_DIS_CE
data_al['NT_GER'] = data_al['NT_GER'].str.replace(',','.')
data_al['NT_GER'] = data_al['NT_GER'].astype(float)

data_al_media = data_al['NT_GER'].mean()

#%% AJUSTE
#data_al.iloc[:,0:16] = data_al.iloc[:,0:16].fillna(data_al.iloc[:,0:16].mean())
data_al['NT_GER'] = data_al['NT_GER'].fillna(data_al_media)
data_al['NT_GER'] = data_al['NT_GER'].replace([0],data_al_media)
# Observando os dados
#print('O formato dos dados é: ', features_al.shape)

describe_al = data_al.describe()

#% 3 - Transformação

# Convertendo os labels de predição para arrays numpy
labels_al = np.array(data_al['NT_GER'])

# Removendo as features de notas
data_al = data_al.drop(['NT_GER'], axis = 1)
'''
data_al = data_al.drop(['NT_GER','NT_FG','NT_OBJ_FG','NT_DIS_FG',
                               'NT_FG_D1','NT_FG_D1_PT','NT_FG_D1_CT',
                               'NT_FG_D2','NT_FG_D2_PT','NT_FG_D2_CT',
                               'NT_CE','NT_OBJ_CE','NT_DIS_CE',
                               'NT_CE_D1','NT_CE_D2','NT_CE_D3'], axis = 1)
'''
# Salvando e convertendo
# Salvando os nomes das colunas (features) com os dados para uso posterior antes de codificar
features_al_list = list(data_al.columns)


# One hot encoding - QE_I01 a QE_I26
features_al = pd.get_dummies(data=data_al, columns=['QE_I01','QE_I02','QE_I03','QE_I04',
                                                        'QE_I05','QE_I06','QE_I07','QE_I08',
                                                        'QE_I09','QE_I10','QE_I11','QE_I12',
                                                        'QE_I13','QE_I14','QE_I15','QE_I16',
                                                        'QE_I17','QE_I18','QE_I19','QE_I20',
                                                        'QE_I21','QE_I22','QE_I23','QE_I24',
                                                        'QE_I25','QE_I26'])
# Salvando os nomes das colunas (features) com os dados para uso posterior
# depois de codificar
features_al_list_oh = list(features_al.columns)
#
# Convertendo para numpy
features_al = np.array(features_al)

#%% TEMPO E AJUSTE
# Tempo de execução
def seconds_transform(seconds_time):
  hours = int(seconds_time/3600)
  rest_1 = seconds_time%3600
  minutes = int(rest_1/60)
  seconds = rest_1 - 60*minutes
  #print(seconds)
  print("Time: ", (hours), "h ", (minutes), "min ", round(seconds,2), " s")

labels_al = np.reshape(labels_al, (-1,1))


#%% SEPARANDO OS DADOS
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import time


train_features_al, test_features_al, train_labels_al, test_labels_al = train_test_split(features_al, labels_al, 
                                                   test_size = 0.333, random_state = 42)

#%% CV: Cross Val Score
n_cv = int(10)

scores_al_rf = []
scores_al_dt = []
scores_al_ls = []

importance_fields_al_rf = 0.0
importance_fields_aux_al_rf = []

importance_fields_al_dt = 0.0
importance_fields_aux_al_dt = []

importance_fields_al_ls = 0.0
importance_fields_aux_al_ls = []

start_time_dt_al = time.time() # Time start

# min_samples_split = 320; min_samples_leaf = 200; max_features= log2

dt_al = DecisionTreeRegressor(min_samples_split=320, min_samples_leaf=200)

accuracy_cv = cross_val_score(dt_al, train_features_al, train_labels_al, cv=n_cv)

dt_al.fit(train_features_al, train_labels_al)

importance_fields_al_dt = dt_al.feature_importances_

seconds_dt_al = (time.time() - start_time_dt_al) # Time end

predict_dt = dt_al.predict(test_features_al)

seconds_transform(seconds_dt_al)
#print("Error: ", round(error_al,2))

accuracy_dt = dt_al.score(test_features_al, test_labels_al)
accuracy_r2 = r2_score(test_labels_al, predict_dt)
accuracy_mse = mean_squared_error(test_labels_al, predict_dt)
accuracy_mae = mean_absolute_error(test_labels_al, predict_dt)

#%% Força bruta
mean_ = np.mean(test_labels_al)
mean_err = np.full((9608,1), mean_, dtype=float)
u = ((labels_al-predict_dt) ** 2).sum()
v = ((labels_al - labels_al.mean())**2).sum()
num_r2 = np.subtract(test_labels_al, predict_dt)
den_r2 = np.subtract(test_labels_al, mean_err)

num_u = u;
den_v = v;
#%% Ao quadrado
num_r2q = np.power(num_r2, 2)
den_r2q = np.power(den_r2, 2)

r2_manual = 1 - (num_r2q.sum()/den_r2q.sum())
r2_uv = 1 - (v/u)
print("R2 força bruta: ", round(r2_manual, 2))
print("R2 força bruta UV: ", round(r2_uv, 8))

#%% ACURÁCIA AL
#print('Accuracy RF: ', round(np.average(scores_al_rf), 2), "%.")
print('Accuracy DT: ', round(np.mean(accuracy_dt), 4))
print('Accuracy R2: ', round(np.mean(accuracy_r2), 4))
print('Accuracy CV: ', round(np.mean(accuracy_cv), 4))
print('Accuracy MSE: ', round(np.mean(accuracy_mse), 4))
print('Accuracy MAE: ', round(np.mean(accuracy_mae), 4))
print("Parameters: ", dt_al.get_params())

importance_fields_al_dt_t = importance_fields_al_dt

#%% VIMP
# Lista de tupla com as variáveis de importância - Árvore de decisão
feature_importances_al_dt = \
[(feature, round(importance, 8)) \
 for feature, importance in zip(features_al_list_oh, importance_fields_al_dt)]

# Print out the feature and importances
# [print('Variable DT: {:20} Importance DT: {}'.format(*pair)) for pair in feature_importances_al_dt];

#%%# GUARDANDO OS VALORES DE VIMP
I01_AL_DT = importance_fields_al_dt_t[0:5]; I02_AL_DT = importance_fields_al_dt_t[5:11]; 

I03_AL_DT = importance_fields_al_dt_t[11:14]; I04_AL_DT = importance_fields_al_dt_t[14:20]; 

I05_AL_DT = importance_fields_al_dt_t[20:26]; I06_AL_DT = importance_fields_al_dt_t[26:32];

I07_AL_DT = importance_fields_al_dt_t[32:40]; I08_AL_DT = importance_fields_al_dt_t[40:47]; 

I09_AL_DT = importance_fields_al_dt_t[47:53]; I10_AL_DT = importance_fields_al_dt_t[53:58]; 

I11_AL_DT = importance_fields_al_dt_t[58:69]; I12_AL_DT = importance_fields_al_dt_t[69:75];

I13_AL_DT = importance_fields_al_dt_t[75:81]; I14_AL_DT = importance_fields_al_dt_t[81:87]; 

I15_AL_DT = importance_fields_al_dt_t[87:93]; I16_AL_DT = importance_fields_al_dt_t[93:94]; 

I17_AL_DT = importance_fields_al_dt_t[94:100]; I18_AL_DT = importance_fields_al_dt_t[100:105]; 

I19_AL_DT = importance_fields_al_dt_t[105:112]; I20_AL_DT = importance_fields_al_dt_t[112:123]; 

I21_AL_DT = importance_fields_al_dt_t[123:125]; I22_AL_DT = importance_fields_al_dt_t[125:130]; 

I23_AL_DT = importance_fields_al_dt_t[130:135]; I24_AL_DT = importance_fields_al_dt_t[135:140];

I25_AL_DT = importance_fields_al_dt_t[140:148]; I26_AL_DT = importance_fields_al_dt_t[148:157];

#%% EXIBINDO OS VALORES DE VIMP
print("I01_AL: ", np.sum(I01_AL_DT))
print("I02_AL: ", np.sum(I02_AL_DT))
print("I03_AL: ", np.sum(I03_AL_DT))
print("I04_AL: ", np.sum(I04_AL_DT))
print("I05_AL: ", np.sum(I05_AL_DT))
print("I06_AL: ", np.sum(I06_AL_DT))
print("I07_AL: ", np.sum(I07_AL_DT))
print("I08_AL: ", np.sum(I08_AL_DT))
print("I09_AL: ", np.sum(I09_AL_DT))
print("I10_AL: ", np.sum(I10_AL_DT))
print("I11_AL: ", np.sum(I11_AL_DT))
print("I12_AL: ", np.sum(I12_AL_DT))
print("I13_AL: ", np.sum(I13_AL_DT))
print("I14_AL: ", np.sum(I14_AL_DT))
print("I15_AL: ", np.sum(I15_AL_DT))
print("I16_AL: ", np.sum(I16_AL_DT))
print("I17_AL: ", np.sum(I17_AL_DT))
print("I18_AL: ", np.sum(I18_AL_DT))
print("I19_AL: ", np.sum(I19_AL_DT))
print("I20_AL: ", np.sum(I20_AL_DT))
print("I21_AL: ", np.sum(I21_AL_DT))
print("I22_AL: ", np.sum(I22_AL_DT))
print("I23_AL: ", np.sum(I23_AL_DT))
print("I24_AL: ", np.sum(I24_AL_DT))
print("I25_AL: ", np.sum(I25_AL_DT))
print("I26_AL: ", np.sum(I26_AL_DT))

#%%
features_al1 = data_al.drop(['QE_I03', 'QE_I07', 'QE_I09', 'QE_I10',
                             'QE_I12','QE_I14', 'QE_I15','QE_I16',
                             'QE_I19', 'QE_I23', 'QE_I24'], axis = 1)

# Salvando os nomes das colunas (features) com os dados para uso posterior antes de codificar
#%% Remodelando
features_al1_list = list(data_al.columns)


# One hot encoding - QE_I01 a QE_I26
features_al1 = pd.get_dummies(data=features_al1, columns=['QE_I01','QE_I02','QE_I04',
                                                     'QE_I05','QE_I06','QE_I08',
                                                     'QE_I11', 'QE_I13','QE_I17',
                                                     'QE_I18', 'QE_I20','QE_I21',
                                                     'QE_I22','QE_I25','QE_I26'])
# Salvando os nomes das colunas (features) com os dados para uso posterior
# depois de codificar
features_al1_list_oh = list(features_al1.columns)
#
# Convertendo para numpy
features_al1 = np.array(features_al1)

#%% Dividindo novamente
train_features_al, test_features_al, \
train_labels_al, test_labels_al = train_test_split(features_al1, labels_al, test_size = 0.33, random_state = 42)

#%% Validando e treinando
accuracy_cv = cross_val_score(dt_al, train_features_al, train_labels_al, cv=n_cv)

dt_al.fit(train_features_al, train_labels_al)

importance_fields_al_dt = dt_al.feature_importances_

predict_dt = dt_al.predict(test_features_al)

#print("Error: ", round(error_al,2))
seconds_transform(seconds_dt_al)
accuracy_dt = dt_al.score(test_features_al, test_labels_al)
accuracy_r2 = r2_score(test_labels_al, predict_dt)
accuracy_mse = mean_squared_error(test_labels_al, predict_dt)
accuracy_mae = mean_absolute_error(test_labels_al, predict_dt)

#%% Força bruta

num_r2 = 

#%% Acurácia AL
#print('Accuracy RF: ', round(np.average(scores_al_rf), 2), "%.")
print('Accuracy DT: ', round(np.average(accuracy_dt), 2))
print('Accuracy R2: ', round(np.average(accuracy_r2), 2))
print('Accuracy MSE: ', round(np.average(accuracy_mse), 2))
print('Accuracy MAE: ', round(np.average(accuracy_mae), 2))
print('Accuracy CV: ', round(np.average(accuracy_cv), 2))
#print('Accuracy LS: ', round(np.average(scores_al_ls), 2), "%.")

#importance_fields_al_rf_t = importance_fields_al_rf/n_cv
#importance_fields_al_dt_t = importance_fields_aux_al_dt #/n_cv
#importance_fields_al_ls_t = importance_fields_al_ls/n_cv

#print('Total RF: ', round(np.sum(importance_fields_al_rf_t),2));
#print('Total DT: ', round(importance_fields_al_dt, 2));
#print('Total LS: ', round(np.sum(importance_fields_al_ls_t),2));
importance_fields_al_dt_t = importance_fields_al_dt

#%% VImp
# Lista de tupla com as variáveis de importância - Árvore de decisão
feature_importances_al_dt = \
[(feature, round(importance, 8)) \
 for feature, importance in zip(features_al_list_oh, importance_fields_al_dt)]
    
#%% Guardando VImp
I01_AL_DT = importance_fields_al_dt_t[0:5]; I02_AL_DT = importance_fields_al_dt_t[5:11]; 

I04_AL_DT = importance_fields_al_dt_t[11:17]; 

I05_AL_DT = importance_fields_al_dt_t[17:23]; I06_AL_DT = importance_fields_al_dt_t[23:29];

I07_AL_DT = importance_fields_al_dt_t[29:37]; I08_AL_DT = importance_fields_al_dt_t[37:44]; 

I09_AL_DT = importance_fields_al_dt_t[44:50]; I10_AL_DT = importance_fields_al_dt_t[50:55]; 

I11_AL_DT = importance_fields_al_dt_t[55:66]; I12_AL_DT = importance_fields_al_dt_t[66:72];

I13_AL_DT = importance_fields_al_dt_t[72:78]; I15_AL_DT = importance_fields_al_dt_t[78:84]; 

I17_AL_DT = importance_fields_al_dt_t[84:90]; I18_AL_DT = importance_fields_al_dt_t[90:95]; 

I19_AL_DT = importance_fields_al_dt_t[95:102]; I20_AL_DT = importance_fields_al_dt_t[102:113]; 

I21_AL_DT = importance_fields_al_dt_t[113:115]; I22_AL_DT = importance_fields_al_dt_t[115:120]; 

I23_AL_DT = importance_fields_al_dt_t[120:125]; I24_AL_DT = importance_fields_al_dt_t[125:130];

I25_AL_DT = importance_fields_al_dt_t[130:138]; I26_AL_DT = importance_fields_al_dt_t[138:147];

#%% Atribuindo
print("I01_AL: ", np.sum(I01_AL_DT))
print("I02_AL: ", np.sum(I02_AL_DT))
print("I04_AL: ", np.sum(I04_AL_DT))
print("I05_AL: ", np.sum(I05_AL_DT))
print("I06_AL: ", np.sum(I06_AL_DT))
print("I07_AL: ", np.sum(I07_AL_DT))
print("I08_AL: ", np.sum(I08_AL_DT))
print("I09_AL: ", np.sum(I09_AL_DT))
print("I10_AL: ", np.sum(I10_AL_DT))
print("I11_AL: ", np.sum(I11_AL_DT))
print("I12_AL: ", np.sum(I12_AL_DT))
print("I13_AL: ", np.sum(I13_AL_DT))
print("I15_AL: ", np.sum(I15_AL_DT))
print("I17_AL: ", np.sum(I17_AL_DT))
print("I18_AL: ", np.sum(I18_AL_DT))
print("I19_AL: ", np.sum(I19_AL_DT))
print("I20_AL: ", np.sum(I20_AL_DT))
print("I21_AL: ", np.sum(I21_AL_DT))
print("I22_AL: ", np.sum(I22_AL_DT))
print("I23_AL: ", np.sum(I23_AL_DT))
print("I24_AL: ", np.sum(I24_AL_DT))
print("I25_AL: ", np.sum(I25_AL_DT))
print("I26_AL: ", np.sum(I26_AL_DT))
