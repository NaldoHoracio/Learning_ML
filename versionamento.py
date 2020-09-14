# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:03:14 2020

@author: edvonaldo
"""

#!/usr/bin/env python
import re
import csv
import pandas as pd
import os.path

fields_version = ['Version']

rows_version = []


if os.path.isfile('Logs/version.csv'):
    file_version_py = "Logs/version.csv"
    
    df = pd.read_csv(r'Logs/version.csv')
    
    teste = df['Version'].iloc[-1]
    
    value = int(teste)
    
    value += 1
    
    #field_version = ['Version']

    rows_version = [[value]]
    
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

    file_version_py = "Logs/version.csv"
    
    field_version = ['Version']
    
    rows_version = [[teste]]
    with open(file_version_py, 'a') as csvfile:
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile, delimiter=';')
        # writing the fields  
        csvwriter.writerow(fields_version)  
        # writing the data rows  
        csvwriter.writerows(rows_version) 
    print ("File not exist")

        
     