import pandas as pd
import numpy as np
import random
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--seed', default=12, type=int, help="seeding")
parser.add_argument('--test_size', default=0.33, type=float, help="ratio of test set")

parser.add_argument('--rf_n_estimators', default=100, type=int, help="n_estimators of Random Forest")

parser.add_argument('--nn_lr', default=1e-3, type=float, help="NN learning rate")
parser.add_argument('--nn_layer1', default=10, type=int, help="NN # of neurals in 1st layer")
parser.add_argument('--nn_layer2', default=2, type=int, help="NN # of neurals in 2nd layer")
parser.add_argument('--nn_max_iter', default=100, type=int, help="NN max iterations")

parser.add_argument('--ngram_upper', default=1, type=int, help="upperbound of n-gram")
parser.add_argument('--tfidf_max_features', default=500, type=int, help="maximun number of features")
parser.add_argument('--tfidf_min_df', default=5, type=int, help="minimum number of example for tf-idf features")

parser.add_argument('--filename', type=str, help="dataset filename csv")
parser.add_argument('--model_name', type=str, default="RF", help='choice of ML model(default: %(default)s)', choices=['RF', 'LR', 'NN'])
parser.add_argument('--exp', type=str, default="test", help='Experiment name')


args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)

exp_folder = "./{}_seed{}".format(args.exp, args.seed)

try:
    os.makedirs(exp_folder)
except:
    pass

print("loading dataset...")
df = pd.read_csv(args.filename)

texts = df['text'].values
labels = df['label'].values


feature_out_file = './processed/{}_tfidf.pkl'.format(args.filename.replace('.csv',''))

if os.path.exists(feature_out_file):
    print("found processed dataset. Loading...")
    features = pickle.load(open(feature_out_file, 'rb'))

else:
    print("processing dataset...")
    tfidf = TfidfVectorizer(min_df=args.tfidf_min_df, 
                            norm='l2', 
                            max_features=args.tfidf_max_features,
                            encoding='latin-1', 
                            ngram_range=(1, args.ngram_upper), 
                            stop_words='english')

    features = tfidf.fit_transform(texts).toarray()
    print(features.shape)

    try:
        os.makedirs('./processed')
    except:
        pass

    print("saving TF-IDF features for ", args.filename)
    pickle.dump(features, open(feature_out_file, 'wb'))


print("splitting to train/test...")
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    labels, 
                                                    test_size=args.test_size, 
                                                    stratify=labels, 
                                                    random_state=args.seed)

if args.model_name == "RF":
    print("training Random Forest model...")
    model = RandomForestClassifier(n_estimators=args.rf_n_estimators, 
                                    class_weight='balanced', 
                                    random_state=args.seed)

elif args.model_name == "LR":
    print("Training Logistic Regression model...")
    model = LogisticRegression(class_weight='balanced',
                                random_state=args.seed)

elif args.model_name == "NN":
    print("Training Feed Forward Neural Network model...")
    model = MLPClassifier(solver='lbfgs', 
                            alpha=args.nn_lr,
                            hidden_layer_sizes=(args.nn_layer1, args.nn_layer2), 
                            random_state=args.seed,
                            max_iter=args.nn_max_iter)

model.fit(X_train, y_train)

print("evaluating...")
preds = model.predict(X_test)
print(classification_report(y_test, preds))


print("save model...")
if args.model_name == "RF":
    model_output_folder = "/RF_nest{}".format(args.rf_n_estimators)

elif args.model_name == "LR":
    model_output_folder = "/LR"

elif args.model_name == "NN":
    model_output_folder = "/NN_lr{}_ly1{}_ly2{}_iter{}".format(args.nn_layer1,
                                                            args.nn_layer1,
                                                            args.nn_layer2,
                                                            args.nn_max_iter)
model_output_folder = exp_folder +  model_output_folder
try:
    os.makedirs(model_output_folder)
except:
    pass

pickle.dump(model, open(model_output_folder + "/model.pkl", 'wb'))

print("save predictions...")
df = pd.DataFrame()
df['truth'] = y_test
df["prediction"] = preds
df.to_csv(model_output_folder + "/prediction_TESTSET.csv")