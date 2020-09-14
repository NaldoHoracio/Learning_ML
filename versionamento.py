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

def version_file(name_file):
    #fields_version = ['Version', 'Teste']
    
    rows_version = []
    
    if os.path.isfile(name_file):
        file_version_py = name_file      
        df = pd.read_csv(name_file, delimiter=';')
        teste = df['Version'].iloc[-1]
        value = int(teste)
        value_ = int(teste)
        value += 1
        value_ += 2
        rows_version = [[value, value_]]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.writer(csvfile, delimiter=';') 
            # writing the fields  
            #csvwriter.writerow(field_version)    
            # writing the data rows  
            csvwriter.writerows(rows_version) 
    else:
        teste = int(0)
        teste = 1
        file_version_py = name_file
        fields_version = ['Version', 'Teste']
        rows_version = [[teste, teste]]
        with open(file_version_py, 'a') as csvfile:
            # creating a csv writer object  
            csvwriter = csv.writer(csvfile, delimiter=';')
            # writing the fields  
            csvwriter.writerow(fields_version)  
            # writing the data rows  
            csvwriter.writerows(rows_version) 
            print ("File not exist")

#%% Return file
version_file(name_file)
     