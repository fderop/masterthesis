# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:41:07 2018

@author: Florian
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.close()
df = pd.read_csv('evolution.csv',names=['year','n'])

sns.set_style("whitegrid", {'axes.grid' : False})
plt.plot(df.year,df.n,linewidth=3)

plt.yticks(np.arange(0, 900, step=100))
plt.xticks(np.arange(2000, 2020, step=2))
plt.xlabel('Year')
plt.ylabel('Number of publications')
sns.despine()
plt.rcParams.update({'font.size': 20})
plt.show()
plt.savefig('evolution.svg', dpi=300)