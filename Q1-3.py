"""
=============generate multivariate gaussian distribution============

 """
 #importing necessary libraries
from scipy import random, linalg
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy
from random import sample
from random import randint
numpy.random.seed(6)
#genertion of covariance matrix and  ensure symmetry and postive semidefinite
matrixSize = 20 
A = numpy.random.random((20, 20))
B = A + A.transpose()
covar_matrix= numpy.dot(B,B.transpose())

#generating mean vectors of class A and class B
mean_classA=numpy.random.rand(20)*2
mean_classB=numpy.random.rand(20)*2
classA=numpy.random.multivariate_normal(mean_classA, covar_matrix, 2000)
classB=numpy.random.multivariate_normal(mean_classB, covar_matrix, 2000)

# percentage overlap
"""
============== randomly selecting train and test datasets=================
"""
#randomly pick 600 data points
test_indices=sample(range(0,2000),600)
test_indices.sort()
test_indices
train_indices=[]
test_incr=0
typeA=numpy.ndarray(shape=(1,1))
typeA[0][0]=1
typeB=numpy.ndarray(shape=(1,1))
typeB[0][0]=0
test=numpy.ndarray(shape=(0,21))
train=numpy.ndarray(shape=(0,21))
for i in range(0,2000):
    curr_rowA=numpy.array([classA[i]])
    curr_rowA=numpy.concatenate((curr_rowA,typeA),axis=1)
    curr_rowB=numpy.array([classB[i]])
    curr_rowB=numpy.concatenate((curr_rowB,typeB),axis=1)
    if test_incr>=600:
        train_indices.append(i)
        train=numpy.concatenate((train,curr_rowA,curr_rowB),axis=0)
    elif (i==test_indices[test_incr]):
        test=numpy.concatenate((test,curr_rowA,curr_rowB),axis=0)
        test_incr=test_incr+1
    else:
        train=numpy.concatenate((train,curr_rowA,curr_rowB),axis=0)
        train_indices.append(i)
#appply LDA'
#pik
X=train[:,0:20]
y=[1,0]*1400
pi1=0.5
pi2=0.5
covar_inv=numpy.linalg.inv(covar_matrix)
output=[0]*1200

"""
==================LDA Model=================================================
"""
for i in range(1200):
    term1= test[i][0:20].dot(covar_inv).dot(mean_classB-mean_classA)
    term2= 0.5*(mean_classB+mean_classA).dot(covar_inv).dot(mean_classB-mean_classA)
    if term1>=term2:
        output[i]=0
    else:
        output[i]=1
real=[1,0]*600

lda_fmeasure=float(2*lda_precision*lda_recall)/(lda_precision+lda_recall)
lda_prec_0=precision_score(real,output,pos_label=0)
lda_prec_1=precision_score(real,output,pos_label=1)
lda_rec_0=recall_score(real,output,pos_label=0)
lda_rec_1=recall_score(real,output,pos_label=1)
lda_fm_0=f1_score(real,output,pos_label=0)
lda_fm_1=f1_score(real,output,pos_label=1)




"""
===============linear classifier======================
"""
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(X,y)
reg_output=reg.predict(test[:,0:20])

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
reg_outputbin=[0]*1200
for i in range(1200):
    if reg_output[i]>0.5:
        reg_outputbin[i]=1

    
lin_prec_0=precision_score(real,reg_outputbin,pos_label=0)
lin_prec_1=precision_score(real,reg_outputbin,pos_label=1)
lin_rec_0=recall_score(real,reg_outputbin,pos_label=0)
lin_rec_1=recall_score(real,reg_outputbin,pos_label=1)
lin_fm_0=f1_score(real,reg_outputbin,pos_label=0)
lin_fm_1=f1_score(real,reg_outputbin,pos_label=1)
lin_acc =accuracy_score(real,reg_outputbin)

"""
==================================knn==========================================
"""
knn_acc_array=[]
knn_prec_array=[]
knn_recall_array=[]
knn_fmeasure_array=[]
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,200):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X,y)
    y_cap=neigh.predict(test[:,0:20])
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(1200):
        if y_cap[i]==1 and real[i]==1:
            tp+=1
        elif y_cap[i]==0 and real[i]==0:
            tn+=1
        elif y_cap[i]==1 and real[i]==0:
            fp+=1
        elif y_cap[i]==0 and real[i]==1:
            fn+=1
    knn_accuracy=(float) (tp+tn)/1200
    knn_precision= (float)(tp)/(tp+fp)
    knn_recall=(float) (tp)/(tp+fn)
    F_measure= (float)(2*tp)/(2*tp+fp+fn)
    knn_acc_array.append(knn_accuracy)
    knn_prec_array.append(knn_precision)
    knn_recall_array.append(knn_recall)
    knn_fmeasure_array.append(F_measure)
import matplotlib.pyplot as plt
plt.plot(range(1,200),knn_fmeasure_array)
plt.xlabel('complexity(k-value)')
plt.ylabel('F-measure')

print "best k is %d" %(numpy.argmax(knn_fmeasure_array))

k=numpy.argmax(knn_fmeasure_array)
neigh=KNeighborsClassifier(n_neighbors=k)
neigh.fit(X,y)
y_cap=neigh.predict(test[:,0:20])
knn_prec_0=precision_score(real,y_cap,pos_label=0)
knn_prec_1=precision_score(real,y_cap,pos_label=1)
knn_rec_0=recall_score(real,y_cap,pos_label=0)
knn_rec_1=recall_score(real,y_cap,pos_label=1)
knn_fm_0=f1_score(real,y_cap,pos_label=0)
knn_fm_1=f1_score(real,y_cap,pos_label=1)
knn_acc =accuracy_score(real,y_cap)