# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from statistics import mode
from nltk.classify import ClassifierI


#reading csv
df = pd.read_csv('fake_or_real_news.csv')
df.head()

y = df.label

df = df.drop('label', axis=1)
print(df.head())
###############################################


###############################################


#Dividing Dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)
##############################################

#method to make text into tfidf format

    
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
    
    


   
    
   
    
    

        
#tfidf_train, y_train = tfidf_method(X_train, X_test)
#training and saving Multinomial classifier 
mn_tfidf_clf = MultinomialNB(alpha=0.1)
mn_tfidf_clf.fit(tfidf_train, y_train)
pred = mn_tfidf_clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("Multinomial Naive Bayes accuracy : %0.3f" % score)
    
save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(mn_tfidf_clf, save_classifier)
save_classifier.close()
    
#training and saving Linear SVC classifier
    
svc_tfidf_clf = LinearSVC()
svc_tfidf_clf.fit(tfidf_train, y_train)
pred = svc_tfidf_clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("Linear Support Vector Classifier accuracy:   %0.3f" % score)
    
save_classifier = open("pickled_algos/LSVC_classifier5k.pickle","wb")
pickle.dump(svc_tfidf_clf, save_classifier)
save_classifier.close()
    
#training and saving Logistic Regression classifier

lg_regr = LogisticRegression()
lg_regr.fit(tfidf_train, y_train)
pred = lg_regr.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("Logistic regression:   %0.3f" % score)
    
save_classifier = open("pickled_algos/LReg_classifier5k.pickle","wb")
pickle.dump(lg_regr, save_classifier)
save_classifier.close()
    
#training and saving Gradient Boost classifier

'''    
gr_boo = GradientBoostingClassifier()
gr_boo.fit(tfidf_train, y_train)
pred = gr_boo.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("Gradient Boost:   %0.3f" % score)
    
save_classifier = open("pickled_algos/Gradient_classifier5k.pickle","wb")
pickle.dump(gr_boo, save_classifier)
save_classifier.close()
'''    
    

#Classifier list
Classifier_List = [ mn_tfidf_clf, svc_tfidf_clf, lg_regr]    
    
    

#feature_p = tfidf_method(X_test)

#Comparing the classifier efficiencies
plt.figure(0).clf()

for model, name in [ 
                     (Classifier_List[0], 'Multinomial Naive Bayes'),
                     (Classifier_List[1], 'Linear Support Vector Classifier'),
                     (Classifier_List[2], 'Logistic regression')]:
                     
    
    
    if 'Multinomial' in name:
        pred = model.predict_proba(tfidf_test)[:,1]
    else: 
        pred = model.decision_function(tfidf_test)
    fpr, tpr, thresh = metrics.roc_curve(y_test.values, pred, pos_label='REAL')
    plt.plot(fpr,tpr,label="{}".format(name))
    plt.xlabel("Number of data points")
    plt.ylabel("Accuracy")
    plt.title("Accruracy Graph on Large dataset")

plt.legend(loc=0)
        
 