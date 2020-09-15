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

rows_version = {'Version': 0, 'Método':'DT', 'Split': 320, 'Leaf': 200, 
                 'Acc': [0.77, 0.85, 0.94], 'Acc médio': 0.85, 'Tempo (h,min,s)': '(0,15,34.7)'}

def version_file(name_file, fields, rows_version):
    #fields_version = ['Version', 'Teste']
    rows_aux = []
    fields_v = []
    
    if os.path.isfile(name_file):
        file_version_py = name_file      
        df = pd.read_csv(name_file)
        teste = df['Version'].iloc[-1]
        value = int(teste)
        value += 1
        rows_version['Version'] = value
        rows_aux = [rows_version]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.DictWriter(csvfile, fieldnames = fields_version) 
            # writing the fields  
            #csvwriter.writerow(field_version)    
            # writing the data rows  
            csvwriter.writerows(rows_aux) 
    else:
        file_version_py = name_file
        rows_aux = [rows_version]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.DictWriter(csvfile, fieldnames = fields_version) 
            # writing the fields
            csvwriter.writeheader()
            # writing the data rows 
            csvwriter.writerows(rows_aux)
            print ("File not exist")

#%% Return file
version_file(name_file, fields_version, rows_version)

#%% Read file

file = pd.read_csv(r'Logs/version.csv')
     