#!/usr/bin/env python
# coding: utf-8

# Q-classify the Size_Categorie using SVM
# 
# month	month of the year: 'jan' to 'dec'
# day	day of the week: 'mon' to 'sun'
# FFMC	FFMC index from the FWI system: 18.7 to 96.20
# DMC	DMC index from the FWI system: 1.1 to 291.3
# DC	DC index from the FWI system: 7.9 to 860.6
# ISI	ISI index from the FWI system: 0.0 to 56.10
# temp	temperature in Celsius degrees: 2.2 to 33.30
# RH	relative humidity in %: 15.0 to 100
# wind	wind speed in km/h: 0.40 to 9.40
# rain	outside rain in mm/m2 : 0.0 to 6.4
# Size_Categorie 	the burned area of the forest ( Small , Large)

# In[140]:


#import libraby
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split ,cross_val_score
import numpy as np


# In[141]:


# load dataset
forest=pd.read_csv("E:\\DATA SCIENCE\\LMS\\ASSIGNMENT\\MY ASSIGNMENT\\SVM\\forestfires.csv")


# In[142]:


# check head of dataset
forest.head(10)


# In[143]:


#check value of count of columns
forest['size_category'].value_counts()


# In[144]:


# cehck all columns name in given Dataset
print(forest.columns)


# In[145]:


#check null value
forest.isna().sum()


# In[146]:


#check value of count of columns
forest['size_category'].value_counts()


# In[147]:


# we need to convert our Target columns using labelencoder
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
forest['size_category']=labelencoder.fit_transform(forest['size_category'])
forest


# In[148]:


#only consider numerical columns
f1=forest.iloc[:,2:10]
f2=forest.iloc[:,30]


# In[149]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[150]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(f1)
df_norm.tail(10)


# In[151]:


#concat both columns
forest3=pd.concat([df_norm,f2],axis=1)
forest3


# In[152]:


#split and difine columns
x=forest3.iloc[:,0:8]
y=forest3.iloc[:,8]


# In[153]:


x


# In[154]:


y


# In[155]:


# perform train,train and test size is 0.35. 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35,stratify=y)


# In[156]:


#perform Gridsearch for decide hyper parameter
clf =SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,0.1,00.1]}]
gsv =GridSearchCV(clf,param_grid,cv=10)
gsv.fit(x_train,y_train)


# In[157]:


#let's find best score of C and gamma
gsv.best_params_,gsv.best_score_


# In[158]:


#insiate above values and make a model
clf = SVC(C= 15,gamma=0.5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test,y_pred)*100
print("Accuracy =",acc)
confusion_matrix(y_test,y_pred)


# In[159]:


#Perform linear as model
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
preds =model_linear.predict(x_test)
print("Accurancy =" ,np.mean(preds==y_test)*100)


# In[160]:


#Perform polynomial as model
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
preds =model_poly.predict(x_test)
print("Accurancy =" ,np.mean(preds==y_test)*100)


# In[161]:


#Perform sigmoid as model
model_s = SVC(kernel = "sigmoid")
model_s.fit(x_train,y_train)
preds =model_s.predict(x_test)
print("Accurancy =" ,np.mean(preds==y_test)*100)


# Q-2
# Prepare a classification model using SVM for salary data 
# 
# Data Description:
# 
# age -- age of a person
# workclass	-- A work class is a grouping of work 
# education	-- Education of an individuals	
# maritalstatus -- Marital status of an individulas	
# occupation	 -- occupation of an individuals
# relationship -- 	
# race --  Race of an Individual
# sex --  Gender of an Individual
# capitalgain --  profit received from the sale of an investment	
# capitalloss	-- A decrease in the value of a capital asset
# hoursperweek -- number of hours work per week	
# native -- Native of an individual
# Salary -- salary of an individual
# 

# In[162]:


#import libraby
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split ,cross_val_score
import numpy as np


# In[163]:


# load dataset
sal_train=pd.read_csv("E:\\DATA SCIENCE\\LMS\\ASSIGNMENT\\MY ASSIGNMENT\\SVM\\SalaryData_Train.csv")
# load dataset
sal_test=pd.read_csv("E:\\DATA SCIENCE\\LMS\\ASSIGNMENT\\MY ASSIGNMENT\\SVM\\SalaryData_Test.csv")


# In[164]:


#check value of count of columns
sal_train['Salary'].value_counts()


# In[165]:


#check value of count of columns
sal_test['Salary'].value_counts()


# In[166]:


#check null value
sal_test.isna().sum()


# In[167]:


#check null value
sal_train.isna().sum()


# In[175]:


string_col=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[176]:


# we need to convert string columns using labelencoder
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
for i in string_col:
    sal_train[i]=label_encoder.fit_transform(sal_train[i])
    sal_test[i]=label_encoder.fit_transform(sal_test[i])


# In[177]:


sal_train.head()


# In[178]:


#Threre are large dataset so split them.
train_x=sal_train.iloc[0:500,0:13]
train_y=sal_train.iloc[0:500,13]
test_x=sal_test.iloc[0:300,0:13]
test_y=sal_test.iloc[0:300,13]


# In[179]:


#SVM Classification using kernels: linear,poly,rbf
from sklearn.svm import SVC


# In[180]:


#Perform linear as model
model_linear = SVC(kernel = "linear")
model_linear.fit(train_x,train_y)
preds =model_linear.predict(test_x)
print("Accurancy =" ,np.mean(preds==test_y)*100)


# In[181]:


#Perform polynomial as model
model_poly = SVC(kernel = "poly")
model_poly.fit(train_x,train_y)
preds =model_poly.predict(test_x)
print("Accurancy =" ,np.mean(preds==test_y)*100)


# In[182]:


#Perform sigmoid as model
model_s = SVC(kernel = "sigmoid")
model_s.fit(train_x,train_y)
preds =model_s.predict(test_x)
print("Accurancy =" ,np.mean(preds==test_y)*100)


# In[187]:


#kernel=rbf
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train_x,train_y)
test_pred_rbf=model_rbf.predict(test_x)
train_rbf_acc=np.mean(train_pred_rbf==train_y)
print("Accurancy =" ,np.mean(test_pred_rbf==test_y)*100)

