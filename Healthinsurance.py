import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import pandas_profiling as pp
from scipy.stats import kurtosis,skew
#dtypes={'sex':'category','smoker':'category','region':'category'}
data= pd.read_csv('insurance.csv')

#print(data)

data=data.drop_duplicates(data.columns).reset_index(drop=True)
#print(data.info(memory_usage='deep'))
#print(data.describe())

df=data.isnull().sum()
#print(df)

plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,square=True,cmap='coolwarm')
#plt.show()
#sns.distplot(data.age, rug=True, rug_kws={"color": "g"},
#	kde_kws={"color": "r", "lw": 3, "label": "KDE"},
#	hist_kws={"histtype": "step", "linewidth": 2,
#	"alpha": 0.5, "color": "g"})

#plt.show()

plt.figure(figsize=(15,10))
#plt.show()
sns.countplot(data.age)
#plt.show()

sns.distplot(data.bmi, rug=True, rug_kws={"color": "g"},kde_kws={"color": "y", "lw": 3, "label": "KDE"},hist_kws={"histtype": "step", "linewidth": 2,"alpha": 0.5, "color": "b"})
#plt.show()


sns.countplot(data.children)
#plt.show()

#4) SMOKER
#print(data.smoker.value_counts())

sns.countplot(data.smoker)
#plt.show()

#5) SEX

#print(data.sex.value_counts())
sns.countplot(data.sex)
#plt.show()


#6) Region

#print(data.region.value_counts())
sns.countplot(data.region)
#plt.show()


#7) Charges

#print(kurtosis(data.charges))
#print(skew(data.charges))


sns.distplot(data.charges, rug=True, rug_kws={"color": "g"},kde_kws={"color": "red", "lw": 3, "label": "KDE"},hist_kws={"histtype": "step", "linewidth": 2,"alpha": 0.3, "color": "y"})

#plt.show()


#Bivariate Analysis

#1) AGE
d1=data[data.columns].corr()['age'][:]
#print(d1)

sns.jointplot(data.age,data.charges,kind='kde')
#plt.show()


#2) BMI

d2=data[data.columns].corr()['bmi'][:]
#print(d2)
sns.regplot(data.bmi,data.charges,color='r',marker='+')

#plt.show()

#Categorical Data

#1) Sex

print(data.sex.value_counts())

data.sex = np.where(data.sex=='male', 0, data.sex)
data.sex = np.where(data.sex=='female', 1, data.sex)

data.sex=data.sex.apply(pd.to_numeric,errors='coerce')




#2) Children

#print(data.children.value_counts())

sns.boxplot(data.children,data.charges)


#df=pd.get_dummies(data.children,drop_first=True)
#data=pd.concat([df,data],axis=1)
#del data['children']



#3) Region
print(data.region.value_counts())

sns.violinplot(data.region,data.charges)
#plt.show()


data.region=data.region.astype('object')
#print(data.region)

'''
df=pd.get_dummies(data.region,drop_first=True)
data=pd.concat([df,data],axis=1)
del data['region']
'''



#4) Smoker
#print(data.smoker.value_counts())

sns.stripplot(data.smoker,data.charges,jitter=True)
#plt.show()

data.smoker = np.where(data.smoker=='no', 0, data.smoker)
data.smoker = np.where(data.smoker=='yes', 1, data.smoker)

data.smoker=data.smoker.apply(pd.to_numeric,errors='coerce')




#print(data.info())

plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
#plt.show()


#Modelling

X= data.iloc[:,:-1].values
y= data.iloc[:,-1].values





from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

import xgboost
from xgboost import XGBRegressor
lr=XGBRegressor()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred_train=lr.predict(X_train)



from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
print(r2_score(y_train,y_pred_train))



