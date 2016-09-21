import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
test=pd.read_csv('test.csv',header=None)
test_labels=pd.read_csv('test_labels.csv',header=None)
train=pd.read_csv('train.csv',header=None)
train_labels=pd.read_csv('train_labels.csv',header=None)
pca = decomposition.PCA(n_components=1)
pca.fit(train)
X=pca.transform(train)
X=pd.DataFrame(X)
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(X,train_labels[0])
x_test=pca.transform(test)
y_cap=lr.predict(x_test)
test_labels['predicted']=y_cap
test_labels['output']=0
test_labels.loc[test_labels['predicted']>=1.5,'output']=2
test_labels.loc[test_labels['predicted']<1.5,'output']=1
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train.iloc[0:1000,0], train.iloc[0:1000,1], train.iloc[0:1000,2], c='blue',label='class 1')
ax.scatter(train.iloc[1000:1999,0], train.iloc[1000:1999,1], train.iloc[1000:1999,2], c='red',label='class 2')
ax.set_xlabel('column1')
ax.set_ylabel('column2')
ax.set_zlabel('column3')
ax.legend(loc='upper left')
#%%
import matplotlib.pyplot as plt
plt.scatter(X,train_labels[0],c=train_labels[0])
plt.axvline(x=0)
plt.xlabel("Projected space")
plt.ylabel('Class label')
#%%
from sklearn.lda import LDA
clf=LDA(n_components=1)
lda_X=clf.fit_transform(train,train_labels)
from sklearn import linear_model
lr=linear_model.LinearRegression()
lr.fit(lda_X,train_labels)
lda_testX=clf.transform(test)
test_labels['LDA']=lr.predict(lda_testX)
test_labels.loc[test_labels['LDA']>=1.5,'LDA_cap']=2
test_labels.loc[test_labels['LDA']<1.5,'LDA_cap']=1
import matplotlib.pyplot as plt
plt.scatter(lda_X,train_labels[0],c=train_labels[0])
plt.axvline(x=0)
plt.xlabel("Projected space")
plt.ylabel('True Class label')
#%%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
f1_score(test_labels[0],test_labels['output'],pos_label=2)
p_accuracy=accuracy_score(test_labels[0],test_labels['output'])
p_prec_1=precision_score(test_labels[0],test_labels['output'],pos_label=1)
p_prec_2=precision_score(test_labels[0],test_labels['output'],pos_label=2)
p_recall_1=recall_score(test_labels[0],test_labels['output'],pos_label=1)
p_recall_2=recall_score(test_labels[0],test_labels['output'],pos_label=2)
p_fscore_1=f1_score(test_labels[0],test_labels['output'],pos_label=1)
p_fscore_2=f1_score(test_labels[0],test_labels['output'],pos_label=2)
l_accuracy=accuracy_score(test_labels[0],test_labels['LDA_cap'])
l_prec_1=precision_score(test_labels[0],test_labels['LDA_cap'],pos_label=1)
l_prec_2=precision_score(test_labels[0],test_labels['LDA_cap'],pos_label=2)
l_recall_1=recall_score(test_labels[0],test_labels['LDA_cap'],pos_label=1)
l_recall_2=recall_score(test_labels[0],test_labels['LDA_cap'],pos_label=2)
l_fscore_1=f1_score(test_labels[0],test_labels['LDA_cap'],pos_label=1)
l_fscore_2=f1_score(test_labels[0],test_labels['LDA_cap'],pos_label=2)
