import pandas as pd
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

seed = 12
test_size = 0.33
RF_n_estimators = 100
ngram_upper = 1
tfidf_max_features = 500
tfidf_min_df = 5
filenane = './news_categorization_reu.csv'

random.seed(seed)
np.random.seed(seed)

print("loading dataset...")
df = pd.read_csv(filenane)

texts = df['text'].values
labels = df['label'].values


print("processing dataset...")
tfidf = TfidfVectorizer(min_df=tfidf_min_df, 
                        norm='l2', 
                        max_features=tfidf_max_features,
                        encoding='latin-1', 
                        ngram_range=(1, ngram_upper), 
                        stop_words='english')

features = tfidf.fit_transform(texts).toarray()
print(features.shape)

print("splitting to train/test...")
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    labels, 
                                                    test_size=test_size, 
                                                    stratify=labels, 
                                                    random_state=seed)

print("training Random Forest model...")
model = RandomForestClassifier(n_estimators=RF_n_estimators, 
                                class_weight='balanced', 
                                random_state=seed)


print("Training Logistic Regression model...")
# model = LogisticRegression(class_weight='balanced',
#                             random_state=seed)


print("Training Feed Forward Neural Network model...")
# model = MLPClassifier(solver='lbfgs', 
#                         alpha=1e-3,
#                         hidden_layer_sizes=(10, 2), 
#                         random_state=seed,
#                         max_iter=100)

model.fit(X_train, y_train)

print("evaluating...")
preds = model.predict(X_test)

print(classification_report(y_test, preds))

