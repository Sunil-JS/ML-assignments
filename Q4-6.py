#%%
"""missing value imputation"""

#identify rows which have missing values
import pandas as pd
df=pd.read_csv("communities.data.csv",header=None)
df.replace('?',numpy.NaN,inplace=True)
features=df.columns.values
cols=df.columns.tolist()
missing_bool=df.isnull().any()
total_misscols=missing_bool.sum()
missing_sum={}
counter=0



#moving all missing columns to end

for i in range(5,128):
    if missing_bool[i] :
        missing_sum[features[i]]=[df.iloc[:,i].isnull().sum(),float(df.iloc[:,i].isnull().sum())/1993]
        cols=cols[:i-counter]+cols[i-counter+1:]+cols[i-counter:i-counter+1]
        #print i
        #print cols
        counter=counter+1

"""
  For monotone missing data
patterns,  either  a  parametric  regression  method  that  as-
sumes  multivariate  normality  or  a  nonparametric  method
that uses propensity scores is appropriate


""" 

header=cols[-23:]
proc_df=df[cols]
X=proc_df.iloc[:,5:-23]
y=proc_df.iloc[:,-23:]
z=y.copy()
z=z.convert_objects(convert_numeric=True)
z=z.fillna(z.mean())
y_ind=proc_df.iloc[:,-23:].isnull()
y_ind=y_ind.astype(int)
"""sample_mean imputation"""
m_imputed=pd.concat((X,z),axis=1)
head_cols=[x for x in range(5,128)]
m_imputed=m_imputed[head_cols]
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
cols=X.columns.tolist()
#cols=cols[0:1]+cols[2:]+cols[1:2]
"""Propesnic score method """
y_cap=numpy.ndarray(shape=(1994,46))
y_class=numpy.ndarray(shape=(1994,23))
#X=X[cols]
for i in range(0,46,2):
    lr.fit( X.iloc[:,:-1], y_ind.iloc[:,i/2])
    y_cap[:,i:i+2]=lr.predict_proba(X.iloc[:,:-1])
    for j in range(1994):
        if y_cap[j,i+1]>0.9:
            y_class[j,i/2]=1
        elif y_cap[j,i+1]>0.7:
            y_class[j,i/2]=2
        elif y_cap[j,i+1]>0.4:
            y_class[j,i/2]=3
        elif y_cap[j,i+1]>0.2:
            y_class[j,i/2]=4
        else:
            y_class[j,i/2]=5
#bootstrap
"""
for a class
 first select all not missing values
 now randomly go on selecting with replacing and fill on missing values
 call it first data set
    
# divide intp 200 groups into
"""
y=y.convert_objects(convert_numeric=True)
y.columns=[x for x in range(23)]
cleaned_y=y.copy()
cleaned_y=cleaned_y.fillna(0)
y_class=pd.DataFrame(y_class)
y_req=pd.concat([y_ind,y_class],axis=1)
y_req.columns=[x for x in range(46)]
y.columns=[x for x in range(23)]
for m in range(10):
    for i in range(1,6):
        y_class_ind=(y_class==i)
        for j in range(23):
            counter=0
            sample1=y_req[y_req[j] ==0].index.tolist()
            sample2=y_req[y_req[j] ==1].index.tolist()
            #print sample2
            sample3=y_req[y_req[j+23==i]].index.tolist()
            sample_a=list(set(sample1).intersection(sample3))
            #print sample_a
            sample_b=list(set(sample2).intersection(sample3))
            #print sample_b
            random.shuffle(sample_b)
            #print sample_b
            while sample_b:
                b=sample_b.pop()
                a=numpy.random.choice(sample_a,replace=True)
                #print cleaned_y.iloc[b,j]
                cleaned_y.iloc[b,j]+=y.iloc[a,j]/10
    #print m
        #print "==============================================="
                
cleaned_y.columns=header
completed_df=pd.concat((X,cleaned_y),axis=1)
col_names=[x for x in range(5,128)]
completed_df=completed_df[col_names]
completed_df.to_csv("completed_df_v2.csv",index=False)
#%%

#%%
"""linear regression"""
comp_df=pd.read_csv('completed_df_v2.csv')
#comp_df=m_imputed.copy()
from sklearn import linear_model
lr=linear_model.LinearRegression()
random.seed(10)
#comp_df.insert(0,'bias',bias)
X=comp_df.iloc[:,:-1]
y=comp_df.iloc[:,-1:]
#del X['communityname string']
org_indices=range(1994)
indices=range(1994)
random.shuffle(indices)
min_residual=9999999
best_beta=numpy.ndarray(shape=(122,1))
sum_r=0
for i in range(0,5):
    test_indices=indices[399*i:((i+1)*399)]
    #print len(test_indices)
    train_indices=set(org_indices)-set(test_indices)
    train_indices=list(train_indices)
    test_indices=list(test_indices)
    X_i=X.iloc[train_indices,:]
    y_i=comp_df.iloc[train_indices,-1:]
    y_testi=y.iloc[test_indices,0].tolist()
    X_test=X.iloc[test_indices,:]
    lr.fit(X_i,y_i)
    y_cap_i=lr.predict(X_test)
    y_comp=pd.DataFrame(columns=['Predicted','real','residual'])
    y_cap_i=pd.DataFrame(y_cap_i)
    y_comp['Predicted']=y_cap_i[0]
    y_comp['real']=y_testi
    y_comp['residual']=(y_comp['Predicted']-y_comp['real'])
    residual_error=(y_comp['residual']**2).sum()
    sum_r+=residual_error
    #print residual_error
    if(min_residual>residual_error):
        min_residual=residual_error
        best_beta=lr.coef_
        intercept=lr.intercept_
        ds2=comp_df.iloc[train_indices,5:]
        ds2_test=comp_df.iloc[test_indices,5:]
        ds2.to_csv("DS2-train.csv",index=False)
        ds2_test.to_csv('DS2-test.csv',index=False)
        
print sum_r
print best_beta

#Ridge regression

comp_df=pd.read_csv('completed_df_v2.csv')
from sklearn import linear_model
X=comp_df.iloc[:,:-1]
y=comp_df.iloc[:,-1:]
#del X['communityname string']
lambd_array=[x for x  in range(1,200,1)]
error={}
for lambd in lambd_array:
        ridge_min_residual=99999
        ridge_best_beta=numpy.ndarray(shape=(121,1))
        residual_error_sum=0
        for i in range(0,5):
            test_indices=indices[399*i:((i+1)*399)]
            #print len(test_indices)
            train_indices=set(org_indices)-set(test_indices)
            train_indices=list(train_indices)
            test_indices=list(test_indices)
            X_i=X.iloc[train_indices,:]
            y_i=y.iloc[train_indices,:]
            y_testi=y.iloc[test_indices,0].tolist()
            X_test=X.iloc[test_indices,:]
            """print "saving files"
            name1='CandC-train%d.csv' %(i+1)
            name2='CandC-test%d.csv' %(i+1)
            ctrain=comp_df.iloc[train_indices,:]
            ctest=comp_df.iloc[test_indices,:]
            ctrain.to_csv(name1,index=False)
            ctest.to_csv(name2,index=False)
            #identity_matrix=numpy.identity(127)"""
            clf = linear_model.Ridge (alpha = 1.0/lambd,normalize=True)
            clf.fit(X_i,y_i)
            y_cap_i=clf.predict(X_test)
            #y_cap_i=pd.DataFrame(y_cap_i)
            y_comp=pd.DataFrame(columns=['Predicted','real','residual'])
            y_cap_i=pd.DataFrame(y_cap_i)
            y_comp['Predicted']=y_cap_i[0]
            y_comp['real']=y_testi
            y_comp['residual']=(y_comp['Predicted']-y_comp['real'])
            residual_error=(y_comp['residual']**2).sum()
            #print residual_error
            if(residual_error<ridge_min_residual):
                ridge_min_residual=residual_error
                best_beta=clf.coef_
                intercept=clf.intercept_
            residual_error_sum+=residual_error/5
            #print residual_error_sum
            
        error[lambd]=[residual_error_sum,ridge_min_residual,best_beta,intercept]
ridge_error=[item[0] for item in error.values()]
import matplotlib.pyplot as plt
plt.plot(lambd_array,ridge_error)
plt.xlabel('complexity factor lambda')
plt.ylabel('avaerage residual error')

opt_lambda=min(error, key=error.get)
#Reduced data set 
features=[0]*122
reduced_data=pd.DataFrame()
for i in range(122):
    if abs(error[opt_lambda][2][0][i])>0.028:
        features[i]=1
        reduced_data[i]=comp_df.iloc[:,i]      
m= sum(features)
#comp_df=m_imputed.copy()
from sklearn import linear_model
lr=linear_model.LinearRegression()
#comp_df.insert(0,'bias',bias)
X=reduced_data.copy()
y=comp_df.iloc[:,-1:]
#del X['communityname string']
min_residual=9999999
best_beta=numpy.ndarray(shape=(m,1))
sum_r=0
for i in range(0,5):
    test_indices=indices[399*i:((i+1)*399)]
    #print len(test_indices)
    train_indices=set(org_indices)-set(test_indices)
    train_indices=list(train_indices)
    test_indices=list(test_indices)
    X_i=X.iloc[train_indices,:]
    y_i=y.iloc[train_indices,-1:]
    y_testi=y.iloc[test_indices,0].tolist()
    X_test=X.iloc[test_indices,:]
    lr.fit(X_i,y_i)
    y_cap_i=lr.predict(X_test)
    y_comp=pd.DataFrame(columns=['Predicted','real','residual'])
    y_cap_i=pd.DataFrame(y_cap_i)
    y_comp['Predicted']=y_cap_i[0]
    y_comp['real']=y_testi
    y_comp['residual']=(y_comp['Predicted']-y_comp['real'])
    residual_error=(y_comp['residual']**2).sum()
    sum_r+=residual_error
    #print residual_error
    if(min_residual>residual_error):
        min_residual=residual_error
        best_beta=lr.coef_
print sum_r
print best_beta