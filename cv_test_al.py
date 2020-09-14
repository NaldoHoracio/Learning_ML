# -*- coding: utf-8 -*-
"""
Teste: Cross Validation AL

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
import time

data_al2014 = pd.read_csv(r'data/AL_2014.csv')
data_al2015 = pd.read_csv(r'data/AL_2015.csv')
data_al2016 = pd.read_csv(r'data/AL_2016.csv')
data_al2017 = pd.read_csv(r'data/AL_2017.csv')
data_al2018 = pd.read_csv(r'data/AL_2018.csv')

#%% Função pré-processing

def pre_processing_set_al(data_al2014, data_al2015, data_al2016, data_al2017, data_al2018):
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
    
    # MERGE NOS DADOS: data al
    frames = [data_al2014, data_al2015, data_al2016, data_al2017, data_al2018];
    data_al = pd.concat(frames);
    
    # Enriquecimento
    data_al['NT_GER'] = data_al['NT_GER'].str.replace(',','.')
    data_al['NT_GER'] = data_al['NT_GER'].astype(float)

    data_al_media = round(data_al['NT_GER'].mean(),2)
    
    data_al['NT_GER'] = data_al['NT_GER'].fillna(data_al_media)
    
    describe_al = data_al.describe()
    
    # 3 - Transformação
    labels_al = np.array(data_al['NT_GER'])

    # Removendo as features de notas
    data_al = data_al.drop(['NT_GER'], axis = 1)
    
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
    
    return features_al, labels_al, features_al_list_oh

#%% Aplicando o pré-processamento
    
features_al, labels_al, features_al_list_oh = pre_processing_set_al(data_al2014, data_al2015, data_al2016, data_al2017, data_al2018)

#%% TEMPO E AJUSTE
# Tempo de execução
def seconds_transform(seconds_time):
  hours = int(seconds_time/3600)
  rest_1 = seconds_time%3600
  minutes = int(rest_1/60)
  seconds = rest_1 - 60*minutes
  #print(seconds)
  print(" ", (hours), "h ", (minutes), "min ", round(seconds,2), " s")
  return hours, minutes, round(seconds,2)

#%% BIBLIOTECAS
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GridSearchCV
import time

train_x_al, test_x_al, train_y_al, test_y_al = train_test_split(features_al, labels_al, test_size=0.33, random_state=42)

#train_x_al = np.array(features_al)
#train_y_al = np.array(labels_al)

#%% CV: Cross Val Score
n_cv = int(5);

scores_al_dt = [];
scores_al_dt_mae = [];
scores_al_dt_mse = [];
#scores_al_dt_mape = [];

scores_al_rf = [];
scores_al_rf_mae = [];
scores_al_rf_mse = [];
#scores_al_rf_mape = [];

scores_al_ls = [];
scores_al_ls_mae = [];
scores_al_ls_mse = [];

importance_fields_al_dt = 0.0;
importance_fields_aux_al_dt = [];

importance_fields_al_rf = 0.0
importance_fields_aux_al_rf = []

importance_fields_al_ls = 0.0
importance_fields_aux_al_ls = []

#%% Árvore de decisão

dt_al = DecisionTreeRegressor(min_samples_split=320, min_samples_leaf=200, random_state=42)

time_dt_al_cv = time.time() # Time start DT CV
# min_samples_split = 320; min_samples_leaf = 200; max_features= log2
accuracy_dt_cv = cross_val_score(dt_al, train_x_al, train_y_al, cv=n_cv, scoring='r2')
sec_dt_al_cv = (time.time() - time_dt_al_cv) # Time end DT CV

print('Accuracy DT CV: ', round(np.mean(accuracy_dt_cv), 4))
seconds_transform(sec_dt_al_cv)

#%% Escrevendo em Arquivo - DT
fields_al_dt = ['Método', 'Split', 'Leaf', 'Acc', 'Acc médio', 'Tempo (h,min,s)']

rows_al_dt = [['DT','320', '200', accuracy_dt_cv, accuracy_dt_cv.mean(),
              seconds_transform(sec_dt_al_cv)]]

file_al_dt = "DT.csv"

with open(file_al_dt, 'a') as csvfile:
    # creating a csv writer object  
    csv_al_dt = csv.writer(csvfile)  
        
    # writing the fields  
    csv_al_dt.writerow(fields_al_dt)  
        
    # writing the data rows  
    csv_al_dt.writerows(rows_al_dt) 

#%% Floresta aleatória

<<<<<<< HEAD
# min_samples_split=20; min_samples_leaf=10

rf_al = RandomForestRegressor(n_estimators=1000, min_samples_split=20, min_samples_leaf=10, random_state=42)
=======
# min_samples_split=40, min_samples_leaf=20
rf_al = RandomForestRegressor(n_estimators=1000, min_samples_split=40, min_samples_leaf=20, random_state=42)
>>>>>>> 6081155c5a3da734e930f742ec0f2a4a110fefcb

time_rf_al_cv = time.time()
accuracy_rf_cv = cross_val_score(rf_al, train_x_al, train_y_al, cv=n_cv, scoring='r2')
#accuracy_rf_cv = GridSearchCV(rf_al, param_grid=param_grid, cv=n_cv)
#grid_rf_al.fit(train_x_al, train_y_al)
#
sec_rf_al_cv = (time.time() - time_rf_al_cv)

#print("Best parameters: ", grid_rf_al)
print('Accuracy RF CV: ', round(np.mean(accuracy_rf_cv), 4))
seconds_transform(sec_rf_al_cv)

#%% Escrevendo em Arquivo - RF
fields_al_rf = ['Método', 'N_tree', 'Split', 'Leaf', 'Acc', 'Acc médio', 'Tempo (h,min,s)']

rows_al_rf = [['RF','10', '40', '20', accuracy_rf_cv, 
               accuracy_rf_cv.mean(), seconds_transform(sec_rf_al_cv)]]

file_al_rf = "RF.csv"

with open(file_al_rf, 'a') as csvfile:
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the fields  
    csvwriter.writerow(fields_al_rf)  
        
    # writing the data rows  
    csvwriter.writerows(rows_al_rf) 

#%% LASSO

ls_al = linear_model.Lasso(alpha=0.005, positive=True, random_state=42)

time_ls_al_cv = time.time()
accuracy_ls_cv = cross_val_score(ls_al, train_x_al, train_y_al, cv=n_cv, scoring='r2')
sec_ls_al_cv = (time.time() - time_ls_al_cv)

print('Accuracy LS CV: ', round(np.mean(accuracy_ls_cv), 4))
seconds_transform(sec_ls_al_cv)

#%% Escrevendo arquivo - LS
fields_al_ls = ['Método', 'Alfa', 'Acc','Acc médio', 'Tempo (h,min,s)']

rows_al_ls = [['LS','0.005', accuracy_ls_cv, accuracy_ls_cv.mean(),
              seconds_transform(sec_ls_al_cv)]]

file_al_ls = "LS.csv"

with open(file_al_ls, 'a') as csvfile:
    # creating a csv writer object  
    csv_al_ls = csv.writer(csvfile)  
        
    # writing the fields  
    csv_al_ls.writerow(fields_al_ls)  
        
    # writing the data rows  
    csv_al_ls.writerows(rows_al_ls) 

#%% Treino dos dados - DT_AL

dt_al = DecisionTreeRegressor(min_samples_split=320, min_samples_leaf=200, random_state=42)

kf_cv_al = KFold(n_splits=n_cv, random_state=42, shuffle=True)

time_dt_al = time.time() # Time start dt loop

for train_index_al, test_index_al in kf_cv_al.split(train_x_al):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_al), '-', np.max(test_index_al))
    
    # Dividindo nas features e labels
    train_features_al = train_x_al[train_index_al]
    test_features_al = train_x_al[test_index_al]
    train_labels_al = train_y_al[train_index_al]
    test_labels_al = train_y_al[test_index_al]
    
    # Ajustando cada features e label com RF e DT
    
    # Método 1 - Árvore de decisão
    
    dt_al.fit(train_features_al, train_labels_al)
    
    predictions_al_dt = dt_al.predict(test_features_al)
    
    accuracy_al_dt = dt_al.score(test_features_al, test_labels_al)

    accuracy_mae_al_dt = mean_absolute_error(test_labels_al, predictions_al_dt)
    
    accuracy_mse_al_dt = mean_squared_error(test_labels_al, predictions_al_dt)
    
    # Importância de variável
    importance_fields_aux_al_dt = dt_al.feature_importances_
    importance_fields_al_dt += importance_fields_aux_al_dt
    
    # Append em cada valor médio
    scores_al_dt.append(accuracy_al_dt)
    
    scores_al_dt_mae.append(accuracy_mae_al_dt)
    
    scores_al_dt_mse.append(accuracy_mse_al_dt)

sec_dt_al = (time.time() - time_dt_al) # Time end dt loop

#seconds_transform(sec_dt_al)
#print("")
seconds_transform(sec_dt_al)

#%% Treino dos dados - RF_AL

rf_al = RandomForestRegressor(n_estimators=1000, min_samples_split=40, min_samples_leaf=20, random_state=42)

time_rf_al = time.time() # Time start dt loop

for train_index_al, test_index_al in kf_cv_al.split(train_x_al):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_al), '-', np.max(test_index_al))
    
    # Dividindo nas features e labels
    train_features_al = train_x_al[train_index_al]
    test_features_al = train_x_al[test_index_al]
    train_labels_al = train_y_al[train_index_al]
    test_labels_al = train_y_al[test_index_al]
    
    # Método 2 - Random Forest
    
    rf_al.fit(train_features_al, train_labels_al)
    
    predictions_al_rf = rf_al.predict(test_features_al)
    
    accuracy_al_rf = rf_al.score(test_features_al, test_labels_al)

    accuracy_mae_al_rf = mean_absolute_error(test_labels_al, predictions_al_rf)
    
    accuracy_mse_al_rf = mean_squared_error(test_labels_al, predictions_al_rf)
     
    # Importância de variável
    importance_fields_aux_al_rf = rf_al.feature_importances_
    importance_fields_al_rf += importance_fields_aux_al_rf
    
    # Append em cada valor médio
    scores_al_rf.append(accuracy_al_rf)
    
    scores_al_rf_mae.append(accuracy_mae_al_rf)
    
    scores_al_rf_mse.append(accuracy_mse_al_rf)

sec_rf_al = (time.time() - time_rf_al) # Time end dt loop

#seconds_transform(sec_rf_al)
#print("")
seconds_transform(sec_rf_al)

#%% Treino dos dados - LS_AL

ls_al = linear_model.Lasso(alpha=0.005, positive=True, random_state=42)

time_ls_al = time.time() # Time start dt loop

for train_index_al, test_index_al in kf_cv_al.split(train_x_al):
    #print("Train index: ", np.min(train_index_al), '- ', np.max(train_index_al))
    print("Test index: ", np.min(test_index_al), '-', np.max(test_index_al))
    
    # Dividindo nas features e labels
    train_features_al = train_x_al[train_index_al]
    test_features_al = train_x_al[test_index_al]
    train_labels_al = train_y_al[train_index_al]
    test_labels_al = train_y_al[test_index_al]
    
    
    # Método 3 - Lasso
    
    ls_al.fit(train_features_al, train_labels_al)
    
    predictions_al_ls = ls_al.predict(test_features_al)
    
    accuracy_al_ls = ls_al.score(test_features_al, test_labels_al)

    accuracy_mae_al_ls = mean_absolute_error(test_labels_al, predictions_al_ls)
    
    accuracy_mse_al_ls = mean_squared_error(test_labels_al, predictions_al_ls)    
    
    # Importância das variáveis
    importance_fields_aux_al_ls = ls_al.coef_
    importance_fields_al_ls += importance_fields_aux_al_ls
    
    # Append em cada valor médio
    scores_al_ls.append(accuracy_al_ls)
    
    scores_al_ls_mae.append(accuracy_mae_al_ls)
    
    scores_al_ls_mse.append(accuracy_mse_al_ls)

sec_ls_al = (time.time() - time_ls_al) # Time end dt loop

#seconds_transform(sec_ls_al_cv)
#print("")
seconds_transform(sec_ls_al)


#%% ACURÁCIA AL
print("1 - Decision Tree")
print('Accuracy AL DT: ', round(np.mean(scores_al_dt), 4))
print('Accuracy MAE AL DT: ', round(np.mean(scores_al_dt_mae), 4))
print('Accuracy MSE AL DT: ', round(np.mean(scores_al_dt_mse), 4))
print("2 - Random Forest")
print('Accuracy AL RF: ', round(np.mean(scores_al_rf), 4))
print('Accuracy MAE AL RF: ', round(np.mean(scores_al_rf_mae), 4))
print('Accuracy MSE AL RF: ', round(np.mean(scores_al_rf_mse), 4))
print("3 - Lasso")
print('Accuracy AL LS: ', round(np.mean(scores_al_ls), 4))
print('Accuracy MAE AL LS: ', round(np.mean(scores_al_ls_mae), 4))
print('Accuracy MSE AL LS: ', round(np.mean(scores_al_ls_mse), 4))

#%% Testando o modelo
scores_al_dt_f = [];
scores_al_dt_r2_f = [];
scores_al_dt_mae_f = [];
scores_al_dt_mse_f = [];
    
predictions_al_dt = dt_al.predict(test_x_al)
    
accuracy_al_dt_f = dt_al.score(test_x_al, test_y_al)

accuracy_mae_al_dt_f = mean_absolute_error(test_y_al, predictions_al_dt)
    
accuracy_mse_al_dt_f = mean_squared_error(test_y_al, predictions_al_dt)
    
importance_fields_al_dt_t = importance_fields_al_dt/n_cv

print('Total VImp DT: ', round(np.sum(importance_fields_al_dt_t),2));

#%% Avaliando o modelo
print('Final Accuracy AL DT: ', round(accuracy_al_dt_f, 4))
print('Final Accuracy MAE AL: ', round(accuracy_mae_al_dt_f, 4))
print('Final Accuracy MSE AL: ', round(accuracy_mse_al_dt_f, 4))

#%% VIMP
# Lista de tupla com as variáveis de importância - Árvore de decisão
feature_importances_al_dt = \
[(feature, round(importance, 4)) \
 for feature, importance in zip(features_al_list_oh, importance_fields_al_dt_t)]

# Print out the feature and importances
[print('Variable DT: {:20} Importance DT: {}'.format(*pair)) for pair in feature_importances_al_dt];

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

#%% Criando dataframe
features_vimp_al_dt = [['I01_AL_DT', np.sum(I01_AL_DT)],['I02_AL_DT', np.sum(I02_AL_DT)],
                    ['I03_AL_DT', np.sum(I03_AL_DT)],['I04_AL_DT', np.sum(I04_AL_DT)],
                    ['I05_AL_DT', np.sum(I05_AL_DT)],['I06_AL_DT', np.sum(I06_AL_DT)],
                    ['I07_AL_DT', np.sum(I07_AL_DT)],['I08_AL_DT', np.sum(I08_AL_DT)],
                    ['I09_AL_DT', np.sum(I09_AL_DT)],['I10_AL_DT', np.sum(I10_AL_DT)],
                    ['I11_AL_DT', np.sum(I11_AL_DT)],['I12_AL_DT', np.sum(I12_AL_DT)],
                    ['I13_AL_DT', np.sum(I13_AL_DT)],['I14_AL_DT', np.sum(I14_AL_DT)],
                    ['I15_AL_DT', np.sum(I15_AL_DT)],['I16_AL_DT', np.sum(I16_AL_DT)],
                    ['I17_AL_DT', np.sum(I17_AL_DT)],['I18_AL_DT', np.sum(I18_AL_DT)],
                    ['I19_AL_DT', np.sum(I19_AL_DT)],['I20_AL_DT', np.sum(I20_AL_DT)],
                    ['I21_AL_DT', np.sum(I21_AL_DT)],['I22_AL_DT', np.sum(I22_AL_DT)],
                    ['I23_AL_DT', np.sum(I23_AL_DT)],['I24_AL_DT', np.sum(I24_AL_DT)],
                    ['I25_AL_DT', np.sum(I25_AL_DT)],['I26_AL_DT', np.sum(I26_AL_DT)],]

df_vimp_al_dt = pd.DataFrame(features_vimp_al_dt, columns=['categoria','valor'])

#%% Ordenando os valores

df_vimp_al_dt = df_vimp_al_dt.sort_values(by='valor', ascending=False)

print("Valores ordenados: ")
print(df_vimp_al_dt)