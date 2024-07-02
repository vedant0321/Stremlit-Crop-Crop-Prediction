
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import pickle

# Load your dataset
dataset = pd.read_csv('E://stramlit//Crop_recommendation.csv')

# Separate features (X) and target (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding the categorical target variable (if necessary)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the classifiers
clf_rf = RandomForestClassifier(n_estimators=10, random_state=0)
clf_gb = GradientBoostingClassifier(n_estimators=10, random_state=0)

# Ensemble method: Voting Classifier
ensemble_clf = VotingClassifier(estimators=[('rf', clf_rf), ('gb', clf_gb)], voting='hard')

# Train the ensemble model
ensemble_clf.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(ensemble_clf, model_file)
