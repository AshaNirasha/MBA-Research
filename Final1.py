# -*- coding: utf-8 -*-
"""
"""

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix

nltk.download('stopwords')
#WordNet is an English dictionary which is a part of Natural Language Tool Kit (NLTK) for Python
nltk.download('wordnet')
#used for tagging words with their parts of speech (POS)
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv(r'D:\MBA\Research Project\Final Defense\Code\Combined1.csv',encoding='mac_roman',header=None,names=['class','text'])
df

#df = pd.read_csv(r'D:\MBA\Research Project\Last\After Interim\Code\Document_1\Combined2.csv',encoding='mac_roman',header=None,names=['class','text'])
#df 

"""# Pre-processing"""

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    rem_num = re.sub('[0-9]+', '', sentence)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)
    # return " ".join(tokens)

df['cleaned_text'] = df.apply(lambda x: preprocess(x.text),axis=1)
df

"""# Feature Extraction"""

def cnt_sentence(x):
  cnt = x.count('.')+x.count('?')+x.count('!')
  if cnt ==0:
    cnt =1
  return cnt


def tot_words(x):
  cnt = len(x.split(' '))
  return cnt

def avg_words(x,sent_cnt):
  cnt = round(tot_words(x)/sent_cnt)
  return cnt

def lexical_diversity(x):
  ld = len(set((x).split(' '))) / tot_words(x)
  return ld
  
#stopwords
def count_stop_words(x):  
  cnt = 0
  for w in x.split(' '):
    if w in stopwords.words('english'):
      cnt=cnt+1
  return cnt

df['#sentences'] =  df['text'].apply(lambda x: cnt_sentence(x))
# df['#sentences'] = np.where(df['#sentences']==0,1,df['#sentences'])

df['Total#words'] =  df['cleaned_text'].apply(lambda x: tot_words(x) )
df['Avg#words'] =  df[['cleaned_text','#sentences']].apply(lambda x: avg_words(x['cleaned_text'],x['#sentences']) , axis=1 )

#Lexical diversity is one aspect of 'lexical richness' and refers to the ratio of different 
   #unique word stems to the total number of words
                                                  
df['lexical_diversity'] =  df['cleaned_text'].apply(lambda x:  lexical_diversity(x)  )

df['Total#dots'] =  df['text'].apply(lambda x: x.count('.'))
df['Total#comma'] =  df['text'].apply(lambda x: x.count(','))
df['Total#semicolon'] =  df['text'].apply(lambda x: x.count(';'))
df['Total#colon'] =  df['text'].apply(lambda x: x.count(':'))
df['Total#Exclamationmark'] =  df['text'].apply(lambda x: x.count('!'))
df['Total#Questionmark'] =  df['text'].apply(lambda x: x.count('?'))
df['Total#Hyphens'] =  df['text'].apply(lambda x: x.count('-'))
df['Total#percentage'] =  df['text'].apply(lambda x: x.count('%'))
df['Total#lessthan'] =  df['text'].apply(lambda x: x.count('<'))
df['Total#greaterthan'] =  df['text'].apply(lambda x: x.count('>'))

df['Avg#dots'] = round(df['Total#dots']/df['#sentences'])
df['Avg#comma'] = round( df['Total#comma']/df['#sentences'])
df['Avg#semicolon'] =  round(df['Total#semicolon']/df['#sentences'])
df['Avg#colon'] =  round(df['Total#colon']/df['#sentences'])
df['Avg#Exclamationmark'] = round( df['Total#Exclamationmark']/df['#sentences'])
df['Avg#Questionmark'] =  round(df['Total#Questionmark']/df['#sentences'])
df['Avg#Hyphens'] = round(df['Total#Hyphens']/df['#sentences'])
df['Avg#percentage'] =  round(df['Total#percentage']/df['#sentences'])
df['Avg#lessthan'] = round( df['Total#lessthan']/df['#sentences'])
df['Avg#greaterthan'] =  round(df['Total#greaterthan']/df['#sentences'])

df['Total#stop_words'] =  df['cleaned_text'].apply(lambda x:  count_stop_words(x)  )
df['Avg#stop_words'] =  round(df['Total#stop_words']/df['#sentences'])


df
df[df['#sentences']==0]
df['text'][2]
df['No'] = df.index
dff = pd.DataFrame()

for index, row in df.iterrows():
  data_tagset = nltk.pos_tag(row['cleaned_text'].split(' '))          #pass by tokens
  df_tagset = pd.DataFrame(data_tagset, columns=['Word', 'Tag'])
  df_tagset = df_tagset.groupby(['Tag']).agg('count')      
  df_tagset_piv = pd.pivot_table(df_tagset, values='Word', index=[],columns=['Tag'], aggfunc=np.sum)
  tag_df = pd.DataFrame(df_tagset_piv)
  

  df1 = df[df['No']==row['No']]
  tag_df['No'] =row['No']
  result = df1.merge(tag_df, how='inner',on='No')
  print(result)

  dff = dff.append(result)

dff = dff.fillna(0)

dff.to_csv(r'D:\MBA\Research Project\Final Defense\Code\final_result1.csv',index=False)

#dff.to_csv(r'D:\MBA\Research Project\Last\After Interim\Code\Document_1\final_result2.csv',index=False)

dff

"""# Model Building"""

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if 'OneClassSVM' in str(classifier):
        predictions = np.where(predictions==-1, 0, predictions)     

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    try:
      print('\nConfusion Matrix :')
      plot_confusion_matrix(classifier,feature_vector_valid, valid_y) 
      plt.show()
    except:
      pass
    
    return metrics.accuracy_score(predictions, valid_y)

dff['class'].value_counts()

dff['class'] = dff['class'].replace('Author',1)
dff['class'] = dff['class'].replace('Web',0)
dff['class'] = dff['class'].replace('RP',0)
dff

dff = dff.drop(['No'],axis=1)
dff.columns
dff['class'].value_counts()


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(dff['cleaned_text'],dff['class'],test_size=0.2,random_state=1)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(dff['cleaned_text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

#see the count_vectorizer as dataframe
xtrain_count_df = pd.DataFrame(xtrain_count.todense(), columns = count_vect.get_feature_names())
xtrain_count_df
# xtrain_count_df[0:1].T[xtrain_count_df[0:1].T[0]>0]

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(dff['cleaned_text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(dff['cleaned_text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3),max_features=5000)
tfidf_vect_ngram_chars.fit(dff['cleaned_text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

"""## Naive Bayes"""

# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print ("NB, Count Vectors: ", accuracy)
print('--------------------------------------------')

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy)
print('--------------------------------------------')

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("NB, N-Gram Vectors: ", accuracy)
print('--------------------------------------------')

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("NB, CharLevel Vectors: ", accuracy)
print('--------------------------------------------')

"""## Logostic Regression Classifier"""

# LLogostic Regression Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print ("LR, Count Vectors: ", accuracy)
print('--------------------------------------------')

# Logostic Regression Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy)
print('--------------------------------------------')

# Logostic Regression Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("LR, N-Gram Vectors: ", accuracy)
print('--------------------------------------------')

# Logostic Regression Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars,train_y, xvalid_tfidf_ngram_chars)
print ("LR, CharLevel Vectors: ", accuracy)
print('--------------------------------------------')

"""## SVM """

# SVM Classifier on Count Vectors
accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print ("SVM, Count Vectors: ", accuracy)
print('--------------------------------------------')


# SVM Classifier on Word Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("SVM, WordLevel TF-IDF: ", accuracy)
print('--------------------------------------------')


# SVM Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("SVM, N-Gram Vectors: ", accuracy)
print('--------------------------------------------')


# SVM Classifier on Character Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("SVM, CharLevel Vectors: ", accuracy)
print('--------------------------------------------')

"""## OneClass SVM"""

from sklearn.svm import OneClassSVM

# One SVM Classifier on Count Vectors
accuracy = train_model( OneClassSVM(gamma='auto'), xtrain_count, train_y, xvalid_count)
print ("One_SVM, Count Vectors: ", accuracy)
print('--------------------------------------------')


# One SVM Classifier on Word Level TF IDF Vectors
accuracy = train_model( OneClassSVM(gamma='auto'), xtrain_tfidf, train_y, xvalid_tfidf)
print ("One_SVM, WordLevel TF-IDF: ", accuracy)
print('--------------------------------------------')



# One SVM Classifier on Ngram Level TF IDF Vectors
accuracy = train_model( OneClassSVM(gamma='auto'), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("One_SVM, N-Gram Vectors: ", accuracy)
print('--------------------------------------------')


# One SVM Classifier on Character Level TF IDF Vectors
accuracy = train_model( OneClassSVM(gamma='auto'), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("One_SVM, CharLevel Vectors: ", accuracy)
print('--------------------------------------------')



"""## My features - SVM"""

dff.head()

dff['lexical_diversity'] = round(dff['lexical_diversity']*100)   #change to int

dff.dtypes

for col in dff.columns:
    if (dff[col].dtypes == 'float64'):
      dff[col] = dff[col].astype('int64')

dff.head()

dff_copy = dff.copy()
y = dff['class']
X = dff_copy.drop(['class','text','cleaned_text'],axis=1)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y,test_size=0.2)
# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(svm.SVC(), train_x, train_y, valid_x)
print ("SVM, derived vectors: ", accuracy)

accuracy = train_model(naive_bayes.MultinomialNB(), train_x, train_y, valid_x)
print ("naive_bayes, derived vectors: ", accuracy)

accuracy = train_model( OneClassSVM(gamma='auto'), train_x, train_y, valid_x)
print ("one_svm, derived vectors: ", accuracy)


#----------------------------------------------------------------------

"""## My features"""

dff.head()

dff['lexical_diversity'] = round(dff['lexical_diversity']*100)   #change to int

dff.dtypes

for col in dff.columns:
    if (dff[col].dtypes == 'float64'):
      dff[col] = dff[col].astype('int64')

dff.head()

dff_copy = dff.copy()
y = dff['class']
X = dff_copy.drop(['class','text','cleaned_text'],axis=1
                  )

x_columns = X.columns
x_columns

from sklearn.metrics import confusion_matrix 
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import OneClassSVM

y = y.replace(0,-1)
y



"""### Apply SMOTE"""

sm = SMOTE(random_state=42)   #imbalancing gives only 48% accuracy
X, y = sm.fit_resample(X, y)

pd.DataFrame(y).value_counts()

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y,test_size=0.2,random_state=42)
classifier = OneClassSVM(gamma='auto')

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions))

"""### Feature selection"""

X = pd.DataFrame(X,columns = x_columns)
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(58,'Score'))  #k =56

X[featureScores.nlargest(3,'Score')['Specs']]   #manual

#select best feature set
accuracy_best = 0
best_no_columns = 0
for i in range(len(X.columns)):
  train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(i+1,'Score')['Specs']], y,test_size=0.2, random_state=42)
  classifier = OneClassSVM(gamma='auto')

  # fit the training dataset on the classifier
  classifier.fit(train_x, train_y)
      
  # predict the labels on validation dataset
  predictions = classifier.predict(valid_x)

  accuracy =  metrics.accuracy_score(predictions, valid_y)
  print(i,' ',accuracy)

  if accuracy > accuracy_best:
    print('best')
    accuracy_best = accuracy
    best_no_columns = i+1
  

print('Best set of columns:',featureScores.nlargest(best_no_columns,'Score'))
print('Best column counts:',best_no_columns)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(best_no_columns,'Score')['Specs']], y,test_size=0.2, random_state=42)
classifier = OneClassSVM(gamma='auto')

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions))

"""### without applyting SMOTE accuracy is high()"""

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y,test_size=0.2,random_state=42)
classifier = OneClassSVM(gamma='auto')

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions))

"""### Feature selection"""

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))

#select best feature set
accuracy_best = 0
best_no_columns = 0
for i in range(len(X.columns)):
  train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(i+1,'Score')['Specs']], y,test_size=0.2, random_state=42)
  classifier = OneClassSVM(gamma='auto')

  # fit the training dataset on the classifier
  classifier.fit(train_x, train_y)
      
  # predict the labels on validation dataset
  predictions = classifier.predict(valid_x)

  accuracy =  metrics.accuracy_score(predictions, valid_y)
  print(i,' ',accuracy)

  if accuracy > accuracy_best:
    print('best')
    accuracy_best = accuracy
    best_no_columns = i+1
  
print('Best set of columns:',featureScores.nlargest(best_no_columns,'Score'))
print('Best column counts:',best_no_columns)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(best_no_columns,'Score')['Specs']], y,test_size=0.2, random_state=42)
classifier = OneClassSVM(gamma='auto')      # Assigning the model  - new born baby it won't have any idea of the environment

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)    #building new model by feeding data   # this is classification model. 
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

len(predictions)
predictions
valid_x
valid_y
#test_x = pd.DataFrame({'lexical_diversity':[8800]})
#classifier.predict(test_x)                                    #give
results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions))


print('--------------------Naive Bayes----------------------------------------------')


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y,test_size=0.2)
classifier = naive_bayes.MultinomialNB()

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions)) 



#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))

#select best feature set
accuracy_best = 0
best_no_columns = 0
for i in range(len(X.columns)):
  train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(i+1,'Score')['Specs']], y,test_size=0.2)
  classifier = naive_bayes.MultinomialNB()

  # fit the training dataset on the classifier
  classifier.fit(train_x, train_y)
      
  # predict the labels on validation dataset
  predictions = classifier.predict(valid_x)

  accuracy =  metrics.accuracy_score(predictions, valid_y)
  print(i,' ',accuracy)

  if accuracy > accuracy_best:
    print('best')
    accuracy_best = accuracy
    best_no_columns = i+1
  

print('Best set of columns:',featureScores.nlargest(best_no_columns,'Score'))
print('Best column counts:',best_no_columns)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(best_no_columns,'Score')['Specs']], y,test_size=0.2)
classifier = naive_bayes.MultinomialNB()      # Assigning the model  - new born baby it won't have any idea of the environment

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)    #building new model by feeding data   # this is classification model. 
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions)) 



print('----------------SVM--------------------------------------------------')


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y,test_size=0.2)
classifier = svm.SVC()

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions)) 



#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))

#select best feature set
accuracy_best = 0
best_no_columns = 0
for i in range(len(X.columns)):
  train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(i+1,'Score')['Specs']], y,test_size=0.2)
  classifier = svm.SVC()

  # fit the training dataset on the classifier
  classifier.fit(train_x, train_y)
      
  # predict the labels on validation dataset
  predictions = classifier.predict(valid_x)

  accuracy =  metrics.accuracy_score(predictions, valid_y)
  print(i,' ',accuracy)

  if accuracy > accuracy_best:
    print('best')
    accuracy_best = accuracy
    best_no_columns = i+1
  

print('Best set of columns:',featureScores.nlargest(best_no_columns,'Score'))
print('Best column counts:',best_no_columns)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(best_no_columns,'Score')['Specs']], y,test_size=0.2)
classifier = svm.SVC()     # Assigning the model  - new born baby it won't have any idea of the environment

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)    #building new model by feeding data   # this is classification model. 
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions)) 


print('----------------LR--------------------------------------------------')


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y,test_size=0.2)
classifier = svm.SVC()

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions)) 



#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))

#select best feature set
accuracy_best = 0
best_no_columns = 0
for i in range(len(X.columns)):
  train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(i+1,'Score')['Specs']], y,test_size=0.2)
  classifier = svm.SVC()

  # fit the training dataset on the classifier
  classifier.fit(train_x, train_y)
      
  # predict the labels on validation dataset
  predictions = classifier.predict(valid_x)

  accuracy =  metrics.accuracy_score(predictions, valid_y)
  print(i,' ',accuracy)

  if accuracy > accuracy_best:
    print('best')
    accuracy_best = accuracy
    best_no_columns = i+1
  

print('Best set of columns:',featureScores.nlargest(best_no_columns,'Score'))
print('Best column counts:',best_no_columns)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X[featureScores.nlargest(best_no_columns,'Score')['Specs']], y,test_size=0.2)
classifier = svm.SVC()     # Assigning the model  - new born baby it won't have any idea of the environment

# fit the training dataset on the classifier
classifier.fit(train_x, train_y)    #building new model by feeding data   # this is classification model. 
    
# predict the labels on validation dataset
predictions = classifier.predict(valid_x)

accuracy =  metrics.accuracy_score(predictions, valid_y)
print(accuracy)

results = confusion_matrix(valid_y, predictions) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',metrics.accuracy_score(valid_y, predictions)) 
print('Report : ')
print(metrics.classification_report(valid_y, predictions)) 


