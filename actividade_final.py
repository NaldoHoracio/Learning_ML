#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:19:09 2020

@author: horacio
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

data = pd.read_csv(r'data/Combo.csv')

#%%
carne = np.asarray(data.Carne)

salsa = np.asarray(data.Salsa)

batatas = data.Papas

refresco = data.Refresco

