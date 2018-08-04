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
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000),figsize=(14,10));


var2 = 'TotalBsmtSF'
data2 = pd.concat([train['SalePrice'], train[var2]], axis=1)
data2.plot.scatter(x=var2, y='SalePrice', ylim=(0,800000),figsize=(14,10))


train['OverallQual'].value_counts()
train['OverallQual'].isnull().sum()

var3 = 'OverallQual'
data3 = pd.concat([train['SalePrice'],train[var3]],axis=1)
f,ax = plt.subplots(figsize=(14,10))
fig = sns.boxplot(x=var3,y='SalePrice',data=data3)
fig.axis(ymin=0,ymax=800000)


train['YearBuilt'].value_counts()
train['YearBuilt'].isnull().sum()

var4 = 'YearBuilt'
data4 = pd.concat([train['SalePrice'],train[var4]],axis=1)
data4.plot.scatter(x=var4,y='SalePrice',ylim=(0,800000),figsize=(14,10))

train['MSZoning'].isnull().sum()


corrmat = train.corr()
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corrmat, vmax=.8, square=True)


k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


sns.set()

cols =['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

sns.pairplot(train[cols],size=2.5)


total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(120)

train.shape

#variables with more than 15% of missing values should be deleted. All Garage variables have around 5% missing
#values which should be deleted. Garagecar variables gives up the most important details about garage so rest of them could be deleted.
#Electrical variable has only one missing value so missing value record is deleted and the variable could be used.



 