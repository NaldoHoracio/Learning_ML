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


# Leitura
path_al2014 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/AL_2014.csv';
path_al2015 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/AL_2015.csv';
path_al2016 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/AL_2016.csv';
path_al2017 = 'C:/Users/edvon/Google Drive/UFAL/TCC/CODES/tcc_codes/read_csv_files/AL_data.csv';
path_al2018 = 'C:/Users/edvon/Google Drive/UFAL/TCC/OutrosDados/AL_2018.csv';


data_al2014 = pd.read_csv(path_al2014)
data_al2015 = pd.read_csv(path_al2015)
data_al2016 = pd.read_csv(path_al2016)
data_al2017 = pd.read_csv(path_al2017)
data_al2018 = pd.read_csv(path_al2018)

# Função pré-processing

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
  print("Time: ", (hours), "h ", (minutes), "min ", round(seconds,2), " s")

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
import time

train_x_al, test_x_al, train_y_al, test_y_al = train_test_split(features_al, labels_al, test_size=0.3, random_state=42)

#%% CV: Cross Val Score
n_cv = int(35);

scores_al_dt = [];
scores_al_dt_r2 = [];
scores_al_dt_mae = [];
scores_al_dt_mse = [];
scores_al_dt_mape = [];
#scores_al_rf = []
#scores_al_ls = []

#importance_fields_al_rf = 0.0
#importance_fields_aux_al_rf = []

importance_fields_al_dt = 0.0;
importance_fields_aux_al_dt = [];

#importance_fields_al_ls = 0.0
#importance_fields_aux_al_ls = []

time_dt_al_cv = time.time() # Time start CV

# min_samples_split = 320; min_samples_leaf = 200; max_features= log2
dt_al = DecisionTreeRegressor(min_samples_split=320, min_samples_leaf=200, random_state=42)

accuracy_cv = cross_val_score(dt_al, train_x_al, train_y_al, cv=n_cv, scoring='r2')

sec_dt_al_cv = (time.time() - time_dt_al_cv) # Time end CV

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
    #rf_al.fit(train_features_al, train_labels_al)
    
    # Método 1 - Árvore de decisão
    
    dt_al.fit(train_features_al, train_labels_al)
    
    predictions_al_dt = dt_al.predict(test_features_al)
    
    accuracy_dt = dt_al.score(test_features_al, test_labels_al)

    accuracy_r2 = r2_score(test_labels_al, predictions_al_dt)

    accuracy_mae = mean_absolute_error(test_labels_al, predictions_al_dt)
    
    accuracy_mse = mean_squared_error(test_labels_al, predictions_al_dt)
    
    
    
    importance_fields_aux_al_dt = dt_al.feature_importances_
    importance_fields_al_dt += importance_fields_aux_al_dt
    
    # Método 2 - Random Forest
    
    # Método 3 - Lasso Regression
    
    #lasso_al.fit(train_features_al, train_labels_al)
    
    # Usando o RF e DT para predição dos dados
    #predictions_al_rf = rf_al.predict(test_features_al)
    #predictions_al_ls = lasso_al.predict(test_features_al)

    # Erro
    #errors_al_rf = abs(predictions_al_rf - test_labels_al)
    #errors_al_dt = abs(predictions_al_dt - test_labels_al)
    #errors_al_ls = abs(predictions_al_ls - test_labels_al)
    
    # Acurácia
    #accuracy_al_rf = 100 - mean_absolute_error(test_labels_al, predictions_al_rf)
    #accuracy_al_dt = 100 - mean_absolute_error(test_labels_al, predictions_al_dt)
    #accuracy_al_ls = 100 - mean_absolute_error(test_labels_al, predictions_al_ls)
    

    #accuracy_mape = ((abs(test_labels_al - predictions_al_dt)/test_labels_al) * 100)
    
    # Importância das variáveis
    #importance_fields_aux_al_rf = rf_al.feature_importances_
    #importance_fields_al_rf += importance_fields_aux_al_rf
    
    
    
    #importance_fields_aux_al_ls = lasso_al.coef_
    #importance_fields_al_ls += importance_fields_aux_al_ls
    
    # Append em cada valor médio
    #scores_al_rf.append(accuracy_al_rf)
    scores_al_dt.append(accuracy_dt)
    scores_al_dt_r2.append(accuracy_r2)
    scores_al_dt_mae.append(accuracy_mae)
    scores_al_dt_mse.append(accuracy_mse)
    #scores_al_dt_mape.append(accuracy_mape)
    #scores_al_ls.append(accuracy_al_ls)

sec_dt_al = (time.time() - time_dt_al) # Time end dt loop

seconds_transform(sec_dt_al_cv)
print("")
seconds_transform(sec_dt_al)
#print("Error: ", round(error_al,2))

#%% ACURÁCIA AL
#print('Accuracy RF: ', round(np.average(scores_al_rf), 2), "%.")
print('Accuracy DT: ', round(np.mean(scores_al_dt), 4))
print('Accuracy CV: ', round(np.mean(accuracy_cv), 4))
print('Accuracy R2: ', round(np.mean(accuracy_r2), 4))
print('Accuracy MAE: ', round(np.mean(accuracy_mae), 4))
print('Accuracy MSE: ', round(np.mean(accuracy_mse), 4))
#print('Accuracy MAPE: ', round(np.mean(accuracy_mape.mean()), 4))
#print("Parameters: ", dt_al.get_params())

#%% Testando o modelo
scores_al_dt_f = [];
scores_al_dt_r2_f = [];
scores_al_dt_mae_f = [];
scores_al_dt_mse_f = [];
scores_al_dt_mape_f = [];

#dt_al.fit(test_x_al, test_y_al)
    
predictions_al_dt = dt_al.predict(test_x_al)
    
accuracy_dt_f = dt_al.score(test_x_al, test_y_al)

accuracy_r2_f = r2_score(test_y_al, predictions_al_dt)

accuracy_mae_f = mean_absolute_error(test_y_al, predictions_al_dt)
    
accuracy_mse_f = mean_squared_error(test_y_al, predictions_al_dt)
    
importance_fields_al_dt_t = dt_al.feature_importances_

print('Total DT: ', round(np.sum(importance_fields_al_dt_t),2));

#%% Avaliando o modelo
print('Final Accuracy DT: ', round(accuracy_dt_f, 4))
print('Final Accuracy R2: ', round(accuracy_r2_f, 4))
print('Final Accuracy MAE: ', round(accuracy_mae_f, 4))
print('Final Accuracy MSE: ', round(accuracy_mse_f, 4))

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