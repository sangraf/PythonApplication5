import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
#data=pd.read_csv('C:\\Users\\User\\source\\repos\\MyDataScient\\possum.csv')
initial_data=pd.read_csv('C:\\Users\\User\\source\\repos\\MyDataScient\\possum.csv')
data=initial_data[['age','hdlngth','skullw','totlngth','taill','footlgth','earconch','eye','chest','belly']]
data=data.dropna()
print(data.head(10))

data.describe()
g=sns.pairplot(data, plot_kws={"s": 10})
g.fig.set_size_inches(6,6)
print(data.describe())
plt.show()

data.corr()
ax = sns.heatmap(data.corr(), annot=True)
plt.show()


'''
print(data)
print(data.isnull().values.any())
data=data.dropna()
print(data.isnull().values.any())
del[data['case']]
print(data.describe())
plt.figure(figsize=(10, 2))
plt.subplot (1, 4, 1)
data['site'].hist()
plt.subplot (1, 4, 2)
data['age'].hist()
plt.subplot (1, 4, 3)
data['hdlngth'].hist()
plt.subplot (1, 4, 4)
data['skullw'].hist()
plt.title('skullw')
plt.show()
boxplot=data.boxplot()
plt.show()
del[data['Pop']]
del[data['sex']]
scaler = StandardScaler()
data_=pd.DataFrame(scaler.fit_transform(data))
boxplot = data_.boxplot()
plt.show()
sex1 = data['sex'].unique()
print(sex1)
m = data[data['sex']=='m']['hdlngth']
f = data[data['sex']=='f']['hdlngth']
stat, p_value = stats.levene(m, f)
print(stat)
print(p_value)
sns.pairplot(data,hue='sex')
'''



