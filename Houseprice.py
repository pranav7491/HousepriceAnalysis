import dask.dataframe as dd
#pip install dask[complete] toolz cloudpickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
%matplotlib inline

train = pd.read_csv("C:/Users/prana/Desktop/Data competition/Salepricetrain.csv")

train.columns
train['SalePrice'].describe()

sns.distplot(train['SalePrice'])

train.shape

train['OverallQual'].head(10)

train['YrSold'].describe()

print("Skewness %f" % train['SalePrice'].skew())

print("Kurtosis %f" % train['SalePrice'].kurt())


train['GrLivArea'].describe()


var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000),figsize=(12,10));


var2 = 'TotalBsmtSF'
data2 = pd.concat([train['SalePrice'], train[var2]], axis=1)
data2.plot.scatter(x=var2, y='SalePrice', ylim=(0,800000),figsize=(12,10))


train['OverallQual'].value_counts()
train['OverallQual'].isnull().sum()

var3 = 'OverallQual'
data3 = pd.concat([train['SalePrice'],train[var3]],axis=1)
f,ax = plt.subplots(figsize=(12,10))
fig = sns.boxplot(x=var3,y='SalePrice',data=data3)
fig.axis(ymin=0,ymax=800000)


train['YearBuilt'].value_counts()
train['YearBuilt'].isnull().sum()

var4 = 'YearBuilt'
data4 = pd.concat([train['SalePrice'],train[var4]],axis=1)
data4.plot.scatter(x=var4,y='SalePrice',ylim=(0,800000))

train['MSZoning'].isnull().sum()


corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


