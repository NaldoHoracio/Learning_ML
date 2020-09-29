# -*- coding: utf-8 -*-
"""
Título: Lendo os arquivos de Log em csv

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

#%% Metrics

metrics_al_dt = pd.read_csv(r'compare_methods/Logs/METRICS_EVALUATE/DT_AL.csv')

metrics_al_rf = pd.read_csv(r'compare_methods/Logs/METRICS_EVALUATE/RF_AL.csv')

metrics_al_ls = pd.read_csv(r'compare_methods/Logs/METRICS_EVALUATE/LS_AL.csv')

metrics_br_dt = pd.read_csv(r'compare_methods/Logs/METRICS_EVALUATE/DT_BR.csv')

metrics_br_rf = pd.read_csv(r'compare_methods/Logs/METRICS_EVALUATE/RF_BR.csv')

metrics_br_ls = pd.read_csv(r'compare_methods/Logs/METRICS_EVALUATE/LS_BR.csv')

#%% Vimp

vimp_al_dt = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_DT_AL.csv')

vimp_al_rf = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_RF_AL.csv')

vimp_al_ls = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_LS_AL.csv')

vimp_br_dt = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_DT_BR.csv')

vimp_br_rf = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_RF_BR.csv')

vimp_br_ls = pd.read_csv(r'compare_methods/Logs/VIMPS/VIMP_LS_BR.csv')