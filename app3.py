import csv
import time
import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
 
class Doc_Classifier:
    X_train=[]
    X_test=[]
    X_predict=[]
    Y_train=[]
    y=[]
    Y1=[]
    size=0
    train_ex=0


    def __init__(self):
        #List to store input text
        data_input=[]
        #List to store output labels
        data_output=[]
        train_text=[]

        

        with open('traintest3.csv','r') as f:
            train_csv=csv.reader(f)
            
            self.size=-1
            for row in train_csv:
                if self.size==-1:
                    self.size=0
                else:
                    self.size=self.size+1
                    data_input.append(row[1])
                    data_output.append(row[0])
        print ('There are ',self.size,' examples in the training set\n')    
          
        #self.train_ex=int(input('Enter the number of examples that should be used to train the model\n'))
        self.train_ex=int(15)  
        #Generate a  permutation to re-shuffle the corpus so that training and testing data can be split randomly          
        perm=np.random.permutation(self.size)
        
        #Shuffle the entire corpus            
        for p in perm:
            train_text.append(data_input[p])
            self.Y_train.append(data_output[p])
        
        self.X_train = np.array(train_text[:self.train_ex])
        self.X_test  = np.array(train_text[self.train_ex:self.size])
        self.Train = np.array(train_text)
        
        
    
        with open('predict.txt') as f:
            lines = f.read().splitlines()
            self.X_predict = np.array(lines)
        
        self.lb=LabelBinarizer()
        self.Y1=self.Y_train[:self.train_ex]
        self.y = self.lb.fit_transform(self.Y1)
        self.y2 = self.lb.fit_transform(self.Y_train)
            

    def SVM_LinearSVCTrain(self):        
        SVM_Classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(LinearSVC()))
                ])
        
         
        SVM_Classifier.fit(self.X_train,self.y)
         
        predicted = SVM_Classifier.predict(self.X_test)
        #y_pred = self.lb.inverse_transform(predicted)
        
        try:
            y_pred = self.lb.inverse_transform(predicted)
            print(predicted)
        except:
            print(predicted)   
          
         
        i=self.train_ex
        correct=0
        for label in y_pred:
            if i > self.Y_train.__len__() -1:
                break
            if label==self.Y_train[i]:
                correct=correct+1
            i = i + 1
        for item, labels in zip(self.X_test, y_pred):
            print('Item: {0} => Label: {1}'.format(item, labels))

        print('Number of Examples used for Training',self.train_ex)
        print('Number of Correctly classified',correct)
        print('Total number of samples classified in Test data',self.size-self.train_ex)
        print('The resulting accuracy using Linear SVC is ',(float(correct)*100/float(self.size-self.train_ex)),'%\n')

        """
        cm=confusion_matrix(self.Y_train[self.train_ex:self.size],y_pred)
        percentage_matrix = 100 * cm / cm.sum(axis=1).astype(float)
        plt.figure(figsize=(16, 16))
        sns.heatmap(percentage_matrix, annot=True,  fmt='.2f', xticklabels=['Python', 'Java', 'Scala'], yticklabels=['Python', 'Java', 'Scala']);
        plt.title('Confusion Matrix (Percentage)');
        #plt.show()
        print(classification_report(self.Y_train[self.train_ex:self.size], y_pred,target_names=['Scala', 'Java', 'Python']))
        """
        
        return y_pred
    
    def SVM_LinearSVC(self):        
        SVM_Classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(LinearSVC()))
                ])
         
        SVM_Classifier.fit(self.Train,self.y2)
         
        predicted = SVM_Classifier.predict(self.X_predict)
        y_pred = self.lb.inverse_transform(predicted)
         
        for item, labels in zip(self.X_predict, y_pred):
            print('Item: {0} => Label: {1}'.format(item, labels))

        
        return y_pred
    
    
"""
Create a giant do-while and then a try catch inside. Use the continue command on error and start over 
"""
start=time.time()
print('Initializing....')
clf=Doc_Classifier()
start=time.time()

print('\nRunning SVM Classification')

clf.SVM_LinearSVCTrain()
clf.SVM_LinearSVC()
time3=time.time()
svm_time=time.time()-start


print('\nThe running time was ',time.time()-start, ' seconds')
    