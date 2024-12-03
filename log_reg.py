import sys
import re
import numpy as np

from numpy import dot
from numpy import add
from numpy.linalg import norm

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from sklearn.metrics import f1_score, accuracy_score


from pyspark import SparkConf,SparkContext
sc = SparkContext.getOrCreate()

import os
 
os.environ['PYSPARK_PYTHON'] = '/lusr/bin/python3.12'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/lusr/bin/python3.12'

if len(sys.argv) != 3:
    print("Usage: log_reg.py <trainDataFile> <testDataFile>", file=sys.stderr)
    exit(-1)

trainFile = sys.argv[1]
testFile = sys.argv[2]

trainPages = sc.textFile(trainFile)
testPages = sc.textFile(testFile)

def get_doc_id_contents(doc):
    # get document ID
    start_id = doc.index('id="') + 4
    end_id = doc.index('"', start_id)
    doc_id = doc[start_id:end_id]
    
    # get document content
    content_start = doc.index('>') + 1
    content_end = doc.index('</doc>', content_start)
    content = doc[content_start:content_end].strip()
    
    return (doc_id, content)

# extract the id and contents from documents
keyAndText = trainPages.map(get_doc_id_contents)

regex = re.compile('[^a-zA-Z]')

# remove all non letter characters
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))


allWords = keyAndListOfWords.flatMap(lambda x: ((word, 1) for word in x[1]))

allCounts = allWords.reduceByKey(add)

topWords = allCounts.top(20000, key=lambda x: x[1])

print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

## TASK 1
print('\nTASK 1:')

topWordsK = sc.parallelize(range(20000))

dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

# this function builds an array for us
def buildArray(listOfIndices):
    listOfIndices = [int(i) for i in listOfIndices if 0 <= i < 20000]

    returnVal = np.zeros(20000)
    
    np.add.at(returnVal, listOfIndices, 1)

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal

dictionary_dict = dict(dictionary.collect())

broadcastDictionary = sc.broadcast(dictionary_dict)

keyAndListOfIndicesTF = keyAndListOfWords.map(lambda x: (x[0], buildArray([broadcastDictionary.value.get(word) for word in x[1] if word in broadcastDictionary.value])))

keyAndListOfIndicesTF.take(1)
listOfIndeces = keyAndListOfIndicesTF.map(lambda x: x[1])
listOfIndeces.cache()

label = keyAndListOfIndicesTF.map(lambda x: 1 if x[0].startswith("AU") else 0)
label.cache()

trainRDD= label.zip(listOfIndeces).map(lambda x: (x[0], np.array(x[1])))
trainRDD.cache()


def llh_cost_gradient(x, coeficients):
    """" LLH - loss function and gradiant """
    theta = np.dot(x[1], coeficients)
    
    cost = - x[0] * theta + np.log(1 + np.exp(theta))
    gradient = - x[1] * x[0] +  x[1] * (np.exp(theta) / (1 + np.exp(theta)))
    
    return cost, gradient


num_iteration = 100
learning_rate = 0.0001
coef = np.ones(20000) 
# number of data dimention, feature vector dimention


cost_array = []
old_cost = 0
old_coef = coef

reg_lambda = 1

for i in range(num_iteration):
    
    rdd = trainRDD.map(lambda x: (llh_cost_gradient(x, coef)))
    result = rdd.reduce(lambda x,y: [x[0] + y[0], x[1] + y[1] ])
    
    gradient = result[1] + 2 * reg_lambda * coef

    cost = result[0] + np.sum(coef)
    
    print(str(i) + " Regression Coef (Weights): "+ str(coef) + ", Cost (negative LLH): " + str(cost))
    
    # update the weights 
    coef = coef - learning_rate * gradient

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def prediction(x, coef, threshold):
    """calculate the theta and label with 1 if theta > 0  """
    theta = np.dot(x[1], coef)
    
    probabilities = sigmoid(theta)
    
    prediction = 1 if probabilities >= threshold else 0
    
    return (x[0] , prediction, probabilities, theta,  x[1])

rdd_results = trainRDD.map(lambda x: prediction(x, coef, 0.5))
coef_word = dictionary.map(lambda x: (x[0], coef[x[1]]))
coef_word_top5 = coef_word.top(5, key=lambda x: x[1])

print(coef_word_top5)

## TASK 2
print('\nTASK 2:')

num_iteration = 100
learning_rate = 0.0001
batch_size = 1024
coef = np.ones(20000) 
# number of data dimention, feature vector dimention


cost_array = []
old_cost = 0
old_coef = coef

reg_lambda = 1

wiki_all = trainRDD.filter(lambda x: x[0] == 0)
wiki_all.cache()
aus_all = trainRDD.filter(lambda x: x[0] == 1)
aus_all.cache()

for i in range(num_iteration):
    wiki_sample = wiki_all.sample(False, batch_size / wiki_all.count(), seed=i)
    aus_sample = aus_all.sample(False, batch_size / aus_all.count(), seed=i)
    sample = wiki_sample.union(aus_sample)
    sample_shuffled = sample.sortBy(lambda _: np.random.rand())
    
    rdd = sample_shuffled.map(lambda x: (llh_cost_gradient(x, coef)))
    result = rdd.reduce(lambda x,y: [x[0] + y[0], x[1] + y[1] ])
    
    gradient = result[1] + 2 * reg_lambda * coef

    cost = result[0] + np.sum(coef)
    
    print("Iteration: " + str(i) + ",  Cost (negative LLH): " + str(cost))
    
    # update the weights 
    coef = coef - learning_rate * gradient

coef_word = dictionary.map(lambda x: (x[0], coef[x[1]]))
coef_word_top5 = coef_word.top(5, key=lambda x: x[1])
print(coef_word_top5)

## TASK 3
print('\nTASK 3:')

trainRDD_LP = trainRDD.map(lambda x: LabeledPoint(x[0], x[1]))

iterations = 100  
reg_param = 0.1 
tol = 1e-6


lr_model_rdd = LogisticRegressionWithLBFGS.train(trainRDD_LP, iterations=iterations, regParam=reg_param, tolerance=tol)

predictions_and_labels = trainRDD_LP.map(lambda point: (float(lr_model_rdd.predict(point.features)), point.label))

correct_predictions = predictions_and_labels.filter(lambda x: x[0] == x[1]).count()
total_data = predictions_and_labels.count()
accuracy = correct_predictions / total_data

print("\n====== Evaluation Results After 100 Iterations ======")
print(f"Model Parameters Used:")
print(f"    - Number of Iterations: {iterations}")
print(f"    - Regularization Parameter (regParam): {reg_param}")
print(f"    - Convergence Tolerance (tol): {tol}\n")

print(f"Model Evaluation After Training:")
print(f"    - Total Data Points: {total_data}")
print(f"    - Correct Predictions: {correct_predictions}")
print(f"    - Accuracy: {accuracy:.4f}\n")


## TASK 4
print("\nTASK 4:")

keyAndText_test = testPages.map(get_doc_id_contents)

keyAndListOfWords_test = keyAndText_test.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))
keyAndListOfIndicesTF_test = keyAndListOfWords_test.map(lambda x: (x[0], buildArray([broadcastDictionary.value.get(word) for word in x[1] if word in broadcastDictionary.value])))

label_test = keyAndListOfIndicesTF_test.map(lambda x: 1 if x[0].startswith("AU") else 0)
label_test.cache()

testRDD = label_test.zip(keyAndListOfIndicesTF_test.map(lambda x: np.array(x[1])))

testRDD_with_docID = keyAndText_test.zip(testRDD)  # (docID, (label, features))

testRDD_LP = testRDD_with_docID.map(lambda x: (x[0], LabeledPoint(x[1][0], x[1][1])))

test_predictions_and_labels = testRDD_LP.map(lambda x: (x[0], x[1].label, float(lr_model_rdd.predict(x[1].features))))

y_pred = test_predictions_and_labels.map(lambda x: x[2]).collect()
y_true = test_predictions_and_labels.map(lambda x: x[1]).collect()

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

false_positives = test_predictions_and_labels.filter(lambda x: x[2] == 1 and x[1] == 0)

false_positive_examples = false_positives.take(3)

print("number of false positives: ", false_positives.count())

print("\n====== False Positive Analysis ======")
for i, ((doc_id, text), actual, predicted) in enumerate(false_positive_examples):
    print(f"\nFalse Positive {i+1} - Document ID: {doc_id}")
    print(f"Text: {text[:1000]}...") # print only the first 1000 characters for analysis
