The package includes the following files:
	
-FluFinderMiner2.py is the mining script that connects to the Twitter API and gathers tweets which are loaded into a CSV file
-app3.py is the SVM algorithm that takes the tweets from the CSV file and the training set CSV to classify them as valid or unvalid, then saves them into results.CSV
-Several additional CSVs and files from previous stages in my thesis progression
-The slides and video from my final presentation

Instructions:

1) Install the required libraries, the system needs the following import statements to import the required libraries:

import glob
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
from sklearn.calibration import (calibration_curve, CalibratedClassifierCV)

2) Obtain Twitter API keys and tokens and add them to the FluFinderMinder2.py app

3) Create a training set for the tweets in the format shown in the existing training file

4) Run app3.py to classify the tweets. The program will first guage the accuracy of the training set and then use the entire training set to build a second model to classify the remaining data

	