import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
data = []
path = '.'
files = [f for f in os.listdir(path) if os.path.isfile(f)]
for f in files:
  with open (f, "r") as myfile:
    data.append(myfile.read())

df = pd.DataFrame(data)
"""
import os
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))
"""
from math import log
import nltk
df_dict={}
i=0
spam_ind=[]
for root,dirs,files in os.walk(".",topdown=True):
    data=[]
    spam_ind=[]
    for names in files:
        location=os.path.join(root,names)
        with open(location,"r") as myfile:
            data.append(myfile.read())
            if "spmsg" in names:
                spam_ind.append(1)
            elif "legit" in names:
                spam_ind.append(0)
            else:
                spam_ind.append(-1)
    if i is not 0 and i is not 1 and i<=11:
        df_dict[i-1]=pd.DataFrame({'content':data,'indicator':spam_ind})
    i=i+1

#tokenize each row of data sets
#df_dict[2].apply(lambda row: nltk.word_tokenize(row['content']), axis=1)
#fdist
for i in range(1,10):
    df_dict[i]['tokenized']=df_dict[i].apply(lambda row: nltk.word_tokenize(row['content']), axis=1)

def getTrain(i,j):
    train=pd.DataFrame(columns=['content','indicator','tokenized'])
    for m in range(1,10):
        if m is not i and m is not j:
            train=pd.concat([train,df_dict[m]],axis=0,ignore_index=True)
    return train
def getTest(i,j):
    test=pd.DataFrame(columns=['content','indicator','tokenized'])
    for m in range(1,10):
        if m is i or m is j:
            test=pd.concat([test,df_dict[m]],axis=0,ignore_index=True)
    return test
      
  
        

def getcondProb(d,b,vocab):
    denom=sum(d.values())+b
    condProb={}
    for key in vocab:
            condProb[key]=float(d[key]+1)/denom
    return condProb
def Multinomial(test,fdist_spam,fdist_legit,vocab,pred_name,score_name):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    b=len(vocab)
    condProb_spam=getcondProb(fdist_spam,b,vocab)
    condProb_legit=getcondProb(fdist_legit,b,vocab)

    ntest_rows=test.shape[0]

    for i in range(ntest_rows):
        spamProb=log(prior_sp)
        legitProb=log(prior_lg)
        for word in test.ix[i,'tokenized']:
            if word in vocab:
                spamProb+=log(condProb_spam[word])
                legitProb+=log(condProb_legit[word])
        test.ix[i,score_name]=legitProb/(spamProb+legitProb)
        if spamProb> legitProb:
            test.ix[i,pred_name]=1
        else:
            test.ix[i,pred_name]=0
    mult_prec=precision_score(test['indicator'],test[pred_name],pos_label=1)
    mult_rec=recall_score(test['indicator'],test[pred_name],pos_label=1)
    mult_fm=f1_score(test['indicator'],test[pred_name],pos_label=1)
    print "precision is %.3f" %(mult_prec)
    print "recall is %.3f" %(mult_rec)
    print "F measure is %.3f" %(mult_fm)
    plotpr(test,'indicator',score_name)
    
    
def plotpr(test,real,score):
    from sklearn.metrics import precision_recall_curve
    m=test.shape[0]
    f_score=[]
    f_real=[]
    for i in range(m):
        if test.ix[i,score] == test.ix[i,score]:
            f_score.append(test.ix[i,score])
            f_real.append(test.ix[i,real])
    precision,recall,thresholds=precision_recall_curve(f_real,f_score,pos_label=1)
    import matplotlib.pyplot as plt
    plt.plot((precision),(recall))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()      
#Multinomial(test,fdist_spam,fdist_legit,vocab,'Mult_pred','Mult_score')
def getcondProbBi(d,vocab,n):
    denom=n+2
    condProb={}
    for key in vocab:
            condProb[key]=float(d[key]+1)/denom
    return condProb 
def Bernouli(test,count_spam,count_legit,vocab,n_spam,n_legit,pred_name,score_name):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    condProb_spamBi=getcondProbBi(count_spam,vocab,n_spam)
    n_legit=n_rows-n_spam
    condProb_legitBi=getcondProbBi(count_legit,vocab,n_legit)
    for i in range(ntest_rows):
        spamProbBi=log(prior_sp)
        legitProbBi=log(prior_lg)
        row=set(test.ix[i,'tokenized'])
        for word in vocab:
            if word in row:
                spamProbBi+=log(condProb_spamBi[word])
                legitProbBi+=log(condProb_legitBi[word])
            else:
                spamProbBi+=log(1-condProb_spamBi[word])
                legitProbBi+=log(1-condProb_legitBi[word])
        """        
        for word in set(test.ix[i,'tokenized']):
            if word in vocab:
                spamProbBi+=log(condProb_spamBi[word])
                legitProbBi+=log(condProb_legitBi[word])
        """
        test.ix[i,score_name]=legitProbBi/(spamProbBi+legitProbBi)
        if spamProbBi> legitProbBi:
            test.ix[i,pred_name]=1
        else:
            test.ix[i,pred_name]=0
    Ber_prec=precision_score(test['indicator'],test[pred_name],pos_label=1)
    Ber_rec=recall_score(test['indicator'],test[pred_name],pos_label=1)
    Ber_fm=f1_score(test['indicator'],test[pred_name],pos_label=1)
    print "precision is %.3f" %(Ber_prec)
    print "recall is %.3f" %(Ber_rec)
    print "F measure is %.3f" %(Ber_fm)
    plotpr(test,'indicator',score_name)
#Bernouli(test,count_spam,count_legit,vocab,n_spam,n_legit,'Ber_pred','Ber_score')


#%%
"""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
precision_score(test['indicator'],test['Prediction'])
precision_score(test['indicator'],test['PredictionBi'])
recall_score(test['indicator'],test['Prediction'])
recall_score(test['indicator'],test['PredictionBi'])
f1_score(test['indicator'],test['PredictionBi'])
f1_score(test['indicator'],test['Prediction'])
"""
def getbayesDir(freq,alpha,vocab):
    denom=0
    beta={}
    for word in vocab:
        denom+=freq[word]+alpha[word]
    for word in vocab:
        beta[word]=float(freq[word]+alpha[word])/(denom)
    return beta
def getbayesBeta(count,alpha,vocab,n):
    denom=n+alpha[0]+alpha[1]
    beta={}
    for word in vocab:
        beta[word]=float(count[word]+alpha[0])/denom
    return beta

def naiveBeta(test,count_spam,count_legit,alpha_beta,vocab):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    beta_spam=getbayesBeta(count_spam,alpha_beta,vocab,n_spam)
    beta_legit=getbayesBeta(count_legit,alpha_beta,vocab,n_legit)
    for i in range(ntest_rows):
        spamProbBi=log(prior_sp)
        legitProbBi=log(prior_lg)
        row=set(test.ix[i,'tokenized'])
        for word in vocab:
            if word in row:
                spamProbBi+=log(beta_spam[word])
                legitProbBi+=log(beta_legit[word])
            else:
                spamProbBi+=log(1-beta_spam[word])
                legitProbBi+=log(1-beta_legit[word])
        """        
        for word in set(test.ix[i,'tokenized']):
            if word in vocab:
                spamProbBi+=log(condProb_spamBi[word])
                legitProbBi+=log(condProb_legitBi[word])
        """
        #print legitProbBi/(legitProbBi+spamProbBi)
        test.ix[i,'Beta_score_spam']=legitProbBi/(legitProbBi+spamProbBi)
        if spamProbBi> legitProbBi:
            test.ix[i,'PredictionBeta']=1
        else:
            test.ix[i,'PredictionBeta']=0
    Beta_prec=precision_score(test['indicator'],test['PredictionBeta'],pos_label=1)
    Beta_rec=recall_score(test['indicator'],test['PredictionBeta'],pos_label=1)
    Beta_fm=f1_score(test['indicator'],test['PredictionBeta'],pos_label=1)
    print "precision is %.3f" %(Beta_prec)
    print "recall is %.3f" %(Beta_rec)
    print "F measure is %.3f" %(Beta_fm)
    plotpr(test,'indicator','Beta_score_spam')
        
def naiveDir(test,fdist_spam,fdist_legit,alpha_dir,vocab):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    dir_spam=getbayesDir(fdist_spam,alpha_dir,vocab)
    dir_legit=getbayesDir(fdist_legit,alpha_dir,vocab)
    for i in range(ntest_rows):
        spamProb=log(prior_sp)
        legitProb=log(prior_lg)
        for word in test.ix[i,'tokenized']:
            if word in vocab:
                spamProb+=log(dir_spam[word])
                legitProb+=log(dir_legit[word])
        test.ix[i,'Dir_score_spam']=legitProb/(spamProb+legitProb)
        if spamProb> legitProb:
            test.ix[i,'PredictionDir']=1
        else:
            test.ix[i,'PredictionDir']=0
    Dir_prec=precision_score(test['indicator'],test['PredictionDir'],pos_label=1)
    Dir_rec=recall_score(test['indicator'],test['PredictionDir'],pos_label=1)
    Dir_fm=f1_score(test['indicator'],test['PredictionDir'],pos_label=1)
    print "precision is %.3f" %(Dir_prec)
    print "recall is %.3f" %(Dir_rec)
    print "F measure is %.3f" %(Dir_fm)
    plotpr(test,'indicator','Dir_score_spam')

#naiveBeta(test,count_spam,count_legit,alpha_beta,vocab)
#naiveDir(test,fdist_spam,fdist_legit,alpha_dir,vocab)

def crossvalidation(df_dict,i,j):
    train=getTrain(i,j)
    test=getTest(i,j)
    global n_rows
    global ntest_rows
    ntest_rows=test.shape[0]
    n_rows=train.shape[0]
    spam_words=[]
    legit_words=[]
    for m in range(n_rows):
        if train.ix[m,'indicator']==1:
            spam_words=spam_words+train.ix[m,'tokenized']
        if train.ix[m,'indicator']==0:
            legit_words=legit_words+train.ix[m,'tokenized']
    fdist_spam=nltk.FreqDist(spam_words)
    fdist_legit=nltk.FreqDist(legit_words)
    del fdist_spam['Subject']
    del fdist_spam[":"]
    del fdist_legit["Subject"]
    del fdist_legit[":"]
    global vocab
    vocab=set(spam_words+legit_words)
    b=len(vocab)
    global n_spam
    n_spam=train['indicator'].sum()
    global n_legit
    n_legit=n_rows-n_spam
    global prior_sp
    prior_sp= float(n_spam)/n_rows
    global prior_lg
    prior_lg= 1-prior_sp
    print"==============================================================="
    print " printing model performance when test set is (%d,%d)" %(i,i+1)
    print"==============================================================="
    print "Multinomial distribution"
    Multinomial(test,fdist_spam,fdist_legit,vocab,'Mult_pred','Mult_score')
    count_spam={}
    count_legit={}
    for word in vocab:
        count_spam[word]=0
        count_legit[word]=0
    for m in range(0,n_rows):
        row=set(train.ix[m,'tokenized'])
        spam=train.ix[m,'indicator']
        if spam:
            for word in row:
                count_spam[word]+=1
        if not spam:
            for word in row:
                count_legit[word]+=1
    print "Bernouli distribution"
    Bernouli(test,count_spam,count_legit,vocab,n_spam,n_legit,'Ber_pred','Ber_score')
    alpha_dir={}
    for word in vocab:
        alpha_dir[word]=0.001
    alpha_beta=[0.0001,0.0001]
    print "Bernouli distrinution with Beta prior"
    naiveBeta(test,count_spam,count_legit,alpha_beta,vocab)
    print "Bernouli distrinution with Dirchlet prior"
    naiveDir(test,fdist_spam,fdist_legit,alpha_dir,vocab)
    
    

    
for i in range(1,10,2):
    crossvalidation(df_dict,i,i+1)