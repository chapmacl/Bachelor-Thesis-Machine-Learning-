import glob
import numpy
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
#from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
#from sklearn.calibration import (calibration_curve, CalibratedClassifierCV)
from sklearn.model_selection._split import RepeatedKFold
from sklearn.metrics import f1_score

 
#read training set from training file 
print('Initiated...') 
df = pd.read_csv("train2.csv", sep=",", encoding="latin-1")

df = df.set_index('id')
df.columns = ['class', 'text']

data = df.reindex(numpy.random.permutation(df.index))
print('Training data read')

#create layout for classifier
pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    ('tfidf',              TfidfTransformer()),
    ('classifier',         OneVsRestClassifier(LinearSVC()))
])

print('Training data...')
#train data 
k_fold = RepeatedKFold(n_splits=6)
#k_fold = KFold(n_splits=6, shuffle = True)
k_fold.get_n_splits(data)

scores = []
for train_indices, test_indices in k_fold.split(data):
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    #Read test data
    files = glob.glob("predict.txt")
    lines = []
    for fle in files:
        with open(fle) as f:
            lines += f.readlines()        
    #test_text = numpy.array(lines)
    print('Test data read')
    #fit training data
    lb = LabelBinarizer()
    Z = lb.fit_transform(train_y)
    
    print('Classifying Test data...')
    #fit test data using results from training
    pipeline.fit(train_text, Z)
    predicted = pipeline.predict(test_text)
    predictions = lb.inverse_transform(predicted)

    #Try to add prediction's probability
    #clf = CalibratedClassifierCV(pipeline)
    #clf.fit(train_text, Z)
    #y_proba = clf.predict_proba(test_text)

    print('Writing results...')
    df2=pd.DataFrame(predictions)
    df2.index+=1
    df2.index.name='Id'
    df2.columns=['Label']
    #df2.to_csv('results.csv',header=True)

    for item, labels in zip(test_text, predictions):
        print('Item: {0} => Label: {1}'.format(item, labels))

    lb=LabelBinarizer()
        
    y = lb.fit_transform(test_y)
    score = f1_score(y, predicted)
    scores.append(score)

print('The resulting accuracy using Linear SVC is ', sum(scores)/len(scores), '%\n')
#print y_proba
"""
percentage_matrix = 100 * cm / cm.sum(axis=1).astype(float)
plt.figure(figsize=(16, 16))
#sns.heatmap(percentage_matrix, annot=True,  fmt='.2f', xticklabels=['Java', 'Python', 'Scala'], yticklabels=['Java', 'Python', 'Scala']);
plt.title('Confusion Matrix (Percentage)');
plt.show()
#print(classification_report(test_y, predictions,target_names=['Java', 'Python', 'Scala'], digits=2))
"""