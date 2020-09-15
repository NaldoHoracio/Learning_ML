# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:03:14 2020

@author: edvonaldo
"""

#!/usr/bin/env python
import re
import os
import csv
import pandas as pd
#import os.path

name_file = os.path.relpath("../Learning_ML/Logs/version.csv")

fields_version = ['Version', 'Método', 'Split', 'Leaf', 'Acc', 'Acc médio', 'Tempo (h,min,s)']

rows_version = [0, 'DT', 320, 200, [0.77, 0.85, 0.94],  0.85, '(0,15,34.7)']

def version_file(name_file, fields, rows_version):
    #fields_version = ['Version', 'Teste']
    
    rows_v = []
    fields_v = []
    
    if os.path.isfile(name_file):
        file_version_py = name_file      
        df = pd.read_csv(name_file, delimiter=';')
        teste = df['Version'].iloc[-1]
        value = int(teste)
        value += 1
        rows_v = [[value, rows_version[1], rows_version[2], 
                   rows_version[3], rows_version[4], 
                   rows_version[5], rows_version[6]]]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.writer(csvfile, delimiter=';') 
            # writing the fields  
            #csvwriter.writerow(field_version)    
            # writing the data rows  
            csvwriter.writerows(rows_v) 
    else:
        #teste = int(0)
        #teste = 1
        file_version_py = name_file
        fields_v = [fields_version[0], fields_version[1], fields_version[2],
                    fields_version[3], fields_version[4], fields_version[5], fields_version[6]]
        rows_v = [[rows_version[0], rows_version[1], rows_version[2], 
                   rows_version[3], rows_version[4], rows_version[5],
                   rows_version[6]]]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.writer(csvfile, delimiter=';')
            # writing the fields  
            csvwriter.writerow(fields_v)  
            # writing the data rows  
            csvwriter.writerows(rows_v) 
            print ("File not exist")

#%% Return file
version_file(name_file, fields_version, rows_version)

#%% Read file

file = pd.read_csv(r'Logs/version.csv', delimiter=';')
     