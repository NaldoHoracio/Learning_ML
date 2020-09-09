#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 18:22:48 2020

@author: horacio
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

data = pd.read_csv(r'data/AlumnosDeDFcultad.csv')

#%%

estatura = np.asarray(data.Estatura)

peso = np.asarray(data.Peso)

sexo = np.asarray(data.Sexo)

mascote = np.asarray(data.Mascota)