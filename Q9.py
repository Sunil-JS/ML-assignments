import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
train=pd.read_csv('Train_features.csv')
test=pd.read_csv('Test_features.csv')
test_labels=pd.read_csv('Test_labels.csv')
train_labels=pd.read_csv('Train_labels.csv')
def process(data,m,n):
    counter=1
    proc_train=np.ndarray(shape=(m,n))
    for k in range(0,n):
        for l in range(0,m):
            proc_train[l][k]=data.iloc[counter,0]
            counter=counter+1
    proc_train=pd.DataFrame(proc_train)
    return proc_train

train=process(train,518,96)

test=process(test,40,96)

lr=LogisticRegression()
lr.fit(train,train_labels)

ds2=pd.concat((train,train_labels),axis=1)
ds2.to_csv('DS2-train.csv')
ds2test=pd.concat((test,test_labels),axis=1)
ds2.to_csv('DS2-test.csv')

test_labels['predicted']=lr.predict(test.iloc[:,:])
test_labels=test_labels.convert_objects(convert_numeric=True)
test_labels.rename(columns={'%%MatrixMarket matrix array real general':'true'}, inplace=True)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
lr_prec_1=precision_score(test_labels['true'],test_labels['predicted'],pos_label=1)
lr_rec_1=recall_score(test_labels['true'],test_labels['predicted'],pos_label=1)
lr_prec_0=precision_score(test_labels['true'],test_labels['predicted'],pos_label=-1)
lr_rec_0=recall_score(test_labels['true'],test_labels['predicted'],pos_label=-1)
lr_fm_1=f1_score(test_labels['true'],test_labels['predicted'],pos_label=1)
lr_fm_0=f1_score(test_labels['true'],test_labels['predicted'],pos_label=-1)

lr1_fm_1=[]
lr1_fm_0=[]
for i in range(1000,10000,100):
    clf=LogisticRegression(penalty='l1',C=1.0/i)
    clf.fit(train,train_labels)
    test_labels['lasso']=clf.predict(test.iloc[:,:])
    test_labels=test_labels.convert_objects(convert_numeric=True)
    lr1_fm_1.append(f1_score(test_labels['true'],test_labels['lasso'],pos_label=1))
    lr1_fm_0.append(f1_score(test_labels['true'],test_labels['lasso'],pos_label=-1))
m=len(lr1_fm_0)
lr1_fm=[]
for i in range(m):
    lr1_fm.append((lr1_fm_0[i]+lr1_fm_1[i])/2)
import matplotlib.pyplot as plt
plt.plot(range(1000,10000,100),lr1_fm)
plt.xlabel('complexity')
plt.ylabel('Fmeasure')
plt.show()