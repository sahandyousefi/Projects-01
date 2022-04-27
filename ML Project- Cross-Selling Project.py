#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report 
import sklearn
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import pickle


# # Dataset Import

# In[5]:


train=pd.read_csv(r'...\train.csv')
test=pd.read_csv(r'...\test.csv')



# # Knowing Columns

# In[7]:

train.columns


# In[9]:


X= train[['Gender', 'Age', 'Driving_License', 'Region_Code','Previously_Insured', 'Vehicle_Age','Vehicle_Damage', 'Annual_Premium','Policy_Sales_Channel', 'Vintage']]
y= train['Response']




# # Seprating categorial and numerical data in columns

# In[11]:


cat=[]
num=[]

for i in train.columns:
    if train[i].dtype=="object":
        cat.append(i)
    else:
        num.append(i)


# In[12]:


print(cat)
print(num)




# # Transforming categorial data into numerical data by "Dummy variables" 

# In[13]:


train =pd.get_dummies(train, columns=cat,drop_first=True)


# In[18]:


X = train.drop(['Response','Driving_License'], axis=1)
y = train['Response']




# ### Since data is imbalanced, we should make data balanced by SMOTE

# In[19]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


# In[20]:


print("Shape of X after over-Sampling:", X.shape)
print("Shape of y after over-Sampling:", y.shape)





# # Splitting data to test and train

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=43)




# # Machine Learning modeling: 
# # First Model: Naive Baye

# In[25]:


NB_m = GaussianNB()
NB_m.fit(X_train, y_train)


# In[26]:


y_train_predict = NB_m.predict(X_train)
model_score = NB_m.score(X_train, y_train)
print(model_score)
print(metrics.confusion_matrix(y_train, y_train_predict))
print(metrics.classification_report(y_train, y_train_predict))




# ### Cross-Validation on Naive Baye

# In[28]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(NB_m, X_train, y_train, cv=10)
scores


# In[29]:


scores = cross_val_score(NB_m, X_test, y_test, cv=10)
scores






# # Second Model: Random Forest

# In[31]:


X_train.dtypes


# In[36]:


random_search = {'criterion': ['entropy', 'gini'],
               'max_depth': [2,3,4,5,6,7,10],
               'min_samples_leaf': [4, 6, 8],
               'min_samples_split': [5, 7,10],
               'n_estimators': [150]}

clf = RandomForestClassifier()
model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 10, 
                               cv = 4, verbose= 1, random_state= 101, n_jobs = -1)
model.fit(X_train,y_train)


# In[39]:


filename = 'rf_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[40]:


filename = 'rf_model.sav'


# In[41]:


rf_load = pickle.load(open(filename, 'rb'))


# # Model Evaluation

# In[44]:


y_pred=model.predict(X_test)


# In[45]:


print (classification_report(y_test, y_pred))


# # ROC Curve & AUC of Random forest classifier¶

# In[54]:


pip install matplotlib


# In[55]:


get_ipython().run_line_magic('pylab', 'inline')


# In[56]:


y_score = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

title('Random Forest ROC curve: CC Fraud')
xlabel('FPR (Precision)')
ylabel('TPR (Recall)')

plot(fpr,tpr)
plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))


# In[57]:


roc_auc_score(y_test, y_score)


# In[ ]:




