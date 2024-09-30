# %%
import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy as np
import random
import gzip
import math

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

# %%
len(dataset)

# %%
answers = {} # Put your answers to each question in this dictionary

# %%
dataset[0]

# %%
### Question 1
np.array(X).reshape(-1, 1)

# %%
def feature(datum):
    # your implementation
    X = [dict['review_text'] for dict in datum]
    exclamations = [i.count('!') for i in X]
    return np.array(exclamations).reshape(-1,1)

# %%
X = feature(dataset)
Y = [dict['rating'] for dict in dataset]

# %%
reg = linear_model.LinearRegression()
reg.fit(X,Y)
theta0 = reg.intercept_
theta1 = reg.coef_[0]
pred = reg.predict(X)
mse = np.mean((Y - pred) ** 2)
print(theta0, theta1, mse)

# %%
answers['Q1'] = [theta0, theta1, mse]

# %%
assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)

# %%
### Question 2

# %%
def feature(datum):
    X = [dict['review_text'] for dict in datum]
    exclamations = [x.count('!') for x in X]
    lengths = [len(x) for x in X]

    return np.column_stack((exclamations, lengths))

# %%
X = feature(dataset)
Y = [dict['rating'] for dict in dataset]

# %%
reg = linear_model.LinearRegression()
reg.fit(X, Y)
theta0 = reg.intercept_
theta1 = reg.coef_[0]
theta2 = reg.coef_[1]
pred = reg.predict(X)
mse = np.mean((Y - pred) ** 2)

print(theta0, theta1, theta2, mse)

# %%
answers['Q2'] = [theta0, theta1, theta2, mse]

# %%
assertFloatList(answers['Q2'], 4)

# %%
### Question 3

# %%
def feature(datum, deg):
    # feature for a specific polynomial degree
    X = [dict['review_text'] for dict in datum]
    exclamations = np.array([x.count('!') for x in X]).reshape(-1, 1)

    if deg == 1:
        return exclamations

    all_polynomials = exclamations
    for i in range(2, deg + 1):
        all_polynomials = np.column_stack((all_polynomials, exclamations ** i))
    
    return all_polynomials

# %%
mses = []
Y = [dict['rating'] for dict in dataset]

for i in range(1, 6):
    X = feature(dataset, i)
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    preds = reg.predict(X)
    mse = np.mean((Y - preds) ** 2)
    mses.append(mse)

mses

# %%
answers['Q3'] = mses

# %%
assertFloatList(answers['Q3'], 5)# List of length 5

# %%
### Question 4

# %%
from sklearn.model_selection import train_test_split

mses = []
Y = [dict['rating'] for dict in dataset]

for i in range(1, 6):
    X = feature(dataset, i)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.50, random_state=42)

    # Linear regression
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    mse = np.mean((y_test - preds) ** 2)
    mses.append(mse)

# %%
answers['Q4'] = mses

# %%
assertFloatList(answers['Q4'], 5)

# %%
### Question 5

# %%
preds = np.mean(y_train)
mae = np.mean(np.abs(y_test - preds))

# %%
answers['Q5'] = mae

# %%
assertFloat(answers['Q5'])

# %%
### Question 6

# %%
f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))

# %%
len(dataset)
dataset[0]

# %%
def feature(datum):
    # your implementation
    X = [dict['review/text'] for dict in datum]
    exclamations = [i.count('!') for i in X]
    return np.array(exclamations).reshape(-1,1)

# %%
X = feature(dataset)
gender = [dict['user/gender'] for dict in dataset]

# convert to numerical
y = [1 if line == 'Female' else 0 for line in y]

# %%
from sklearn.metrics import confusion_matrix

reg = linear_model.LogisticRegression()
reg.fit(X, y)
preds = reg.predict(X)


TN, FP, FN, TP = confusion_matrix(y, preds).ravel()
BER = 1/2 * ( (FN/(TP+FN)) + (FP/(TN+FP)) )
print(TN, FP, FN, TP, BER)


# %%
answers['Q6'] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q6'], 5)

# %%
### Question 7

# %%
reg = linear_model.LogisticRegression(class_weight='balanced')
reg.fit(X, y)
preds = reg.predict(X)


TN, FP, FN, TP = confusion_matrix(y, preds).ravel()
BER = 1/2 * ( (FN/(TP+FN)) + (FP/(TN+FP)) )
print(TN, FP, FN, TP, BER)

# %%
answers["Q7"] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q7'], 5)

# %%
### Question 8

# %%
probabilities = reg.predict_proba(X)[:, 1]
sorted_indices = np.argsort(-probabilities)
K = [1, 10, 100, 1000, 10000]
precisionList = []

# %%
from sklearn.metrics import precision_score

for k in K:
    k_num_indices = sorted_indices[:k]
    y_k = np.array(y)[k_num_indices]
    preds_k = np.array(preds)[k_num_indices]
    precisionList.append(precision_score(y_k, preds_k))

# %%
answers['Q8'] = precisionList

# %%
assertFloatList(answers['Q8'], 5) #List of five floats

# %%
f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()

# %%



