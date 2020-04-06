import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import os
import time
data_train = sys.argv[1]
data_test = sys.argv[2]
train = pd.read_csv(data_train)
test = pd.read_csv(data_test)
lamb = 0.01
stepsize = 0.01
max_itr_count =500
stepsize_svm = 0.5
method = sys.argv[3]
model = int(method)
def prediction_lr(data,w_new):
    df = pd.DataFrame(data[data.columns.drop('decision')])
    predict = 1 / (1 + np.exp(-(np.dot(w_new,df.T))))
    predict = np.where(predict >= 0.5, 1, 0)
    return predict
def learnlr(train):
    learn_train = train
    learn_train['intercept'] = 1
    w_old = np.zeros(len(learn_train.columns)-1, dtype=int)
    w_new = np.zeros(len(w_old), dtype=int)
    iteration_count = 0;   
    while (iteration_count < max_itr_count):
        if abs(w_old-w_new).any > (1e-6):
            w_old = w_new
            df = pd.DataFrame(learn_train[learn_train.columns.drop('decision')])
            np_df = df.as_matrix()
            predict = 1 / (1 + np.exp(-(np.dot(w_old,np_df.T))))
            #predict = (predict>=0.5).astype(int)
            predict = np.where(predict >= 0.5, 1, 0) 
            diff = (-learn_train['decision']+predict)
            grad = np.dot(np_df.T,diff) + lamb*w_old
            w_new = w_old - stepsize* grad
            iteration_count +=1
        else:
            break   
    return w_new,learn_train

def printaccuracy_lr(decision,predict):
    accuracy = abs(decision-predict)
    accuracy =  round(float((accuracy==0).sum())/((accuracy==0).sum()+(accuracy==1).sum()),2)
    return accuracy

def prediction_svm(data,w_new):
    df = pd.DataFrame(data[data.columns.drop('decision')])
    predict = np.dot(w_new,df.T)
    predict = np.where(predict >= 0, 1, -1)
    return predict
#function to intitially set weight vectors old,new to 0 and the run a loop to calculate gradient 
#descent. retur new weight vectors in the end.
def learnsvm(train):
    svm_train = train
    svm_train['intercept'] = 1
    w_old = np.zeros(len(svm_train.columns)-1, dtype=int)
    w_new = np.zeros(len(w_old), dtype=int)
    iteration_count = 0;
    while (iteration_count<500):
        if abs(w_old-w_new).all > (1e-6):     
            w_old=w_new
            df = pd.DataFrame(svm_train[svm_train.columns.drop('decision')])
            np_df = df.as_matrix()
            predict = np.dot(w_old,np_df.T)
            predict = np.where(predict >= 0, 1, -1) 
            diff = (svm_train['decision']*predict)
            B = svm_train[(svm_train['decision'] == 1) & (predict == 1)]
            dec = B['decision']
            B.drop('decision',axis =1,inplace=True)
            gradient = np.dot(dec,B)
            grad_intercept = gradient[len(gradient)-1]/len(svm_train)
            grad_final = lamb*w_old[:len(w_old)-1] - gradient[:len(gradient)-1]/len(svm_train)
            grad_intercept = np.array(grad_intercept)
            grad_final = np.append(grad_final,grad_intercept)
            w_new = w_old - stepsize_svm*grad_final
            iteration_count +=1
        else:
            break
    return w_new,svm_train
def printaccuracy_svm(decision,prediction):
    accuracy = np.where((decision & prediction), 1,0)
    accuracy =  round(float((accuracy==0).sum())/((accuracy==0).sum()+(accuracy==1).sum()),2)
    return accuracy

if model == 1:
    w_new,learn_train = learnlr(train) 
    predict = prediction_lr(learn_train,w_new)
    train_accuracy = printaccuracy_lr(train['decision'],predict)
    print "Training Accuracy LR:",train_accuracy
    test['intercept'] = 1
    predict_test = prediction_lr(test,w_new)
    test_accuracy = printaccuracy_lr(test['decision'],predict_test)
    print "Test Accuracy LR:",test_accuracy
elif model == 2:
    w_news,svm_train = learnsvm(train) 
    predict = prediction_svm(svm_train,w_news)
    train_accuracy = printaccuracy_svm(train['decision'],predict)
    print "Training Accuracy SVM:",train_accuracy
    test['intercept'] = 1
    predict_test = prediction_svm(test,w_news)
    test_accuracy = printaccuracy_svm(test['decision'],predict_test)
    print "Test Accuracy SVM:",test_accuracy
else:
    print "Nope!Nope!You are asking me something I don't know?!"

