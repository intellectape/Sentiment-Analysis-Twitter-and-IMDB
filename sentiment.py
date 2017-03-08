# This file is created by:
# Name: Aditya Bhardwaj
# Unity Id: abhardw2
# Project 3: Supervised Learning Techniques for Sentiment Analytics

import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
stopwords = nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print("python sentiment.py <path_to_data> <0|1>")
    print("0 = NLP, 1 = Doc2Vec")
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print("Naive Bayes")
    print("-----------")
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print("")
    print("Logistic Regression")
    print("-------------------")
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

def createDictionary(inputs): 
    dictionary = dict()

    for index in inputs:
        for item in set(index):
            if item in dictionary.keys():
                dictionary[item] += 1
            else:
                dictionary[item] = 1
    return dictionary

def constructVector(vector, features):
    vectorList = []
    for item in vector:
        binarylist = [0]*len(features)
        for word in item:
            if word in features:
                binarylist[features.index(word)] = 1
        vectorList.append(binarylist)
    return vectorList

def removeStopWords(words, stopwords): 
    i =0
    for item in words:
        words[i] = list(set(item) - stopwords)
        i += 1
    return words

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    train_pos = removeStopWords(train_pos, stopwords)
    train_neg = removeStopWords(train_neg, stopwords)

    positive_words = createDictionary(train_pos)
    negative_words = createDictionary(train_neg)
    
    refinedSet = []
    for word, count in positive_words.items():
                if (count >= len(train_pos)*0.01) or (count >= len(train_neg)*0.01):
                        refinedSet.append(word)
    for word, count in negative_words.items():
                if (count >= len(train_pos)*0.01) or (count >= len(train_neg)*0.01):
                        refinedSet.append(word)

    finalList = []
    for word in list(set(refinedSet)):
        if 2*positive_words[word] <= negative_words[word] or 2*negative_words[word] <= positive_words[word]:
            finalList.append(word)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    train_pos_vec = constructVector(train_pos, finalList)
    train_neg_vec = constructVector(train_neg, finalList)
    test_pos_vec = constructVector(test_pos, finalList)
    test_neg_vec = constructVector(test_neg, finalList)
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    labeled_train_pos, labeled_train_neg, labeled_test_pos, labeled_test_neg = [], [], [], []
    index = 0
    for labels in train_pos:
        labeled_train_pos.append(LabeledSentence(labels, ['TRAIN_POS_' + str(index)]))
        index += 1
    
    index = 0
    for labels in train_neg:
        labeled_train_neg.append(LabeledSentence(labels, ['TRAIN_NEG_' + str(index)]))
        index += 1

    index = 0
    for labels in test_pos:
        labeled_test_pos.append(LabeledSentence(labels, ['TEST_POS_' + str(index)]))
        index += 1

    index = 0
    for labels in test_neg:
        labeled_test_neg.append(LabeledSentence(labels, ['TEST_NEG_' + str(index)]))
        index += 1

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print("Training iteration %d"%(i))
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE

    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = [], [], [], []

    for i in range(len(train_pos)):
        train_pos_vec.append(model.docvecs['TRAIN_POS_' + str(i)])
    for i in range(len(train_neg)): 
        train_neg_vec.append(model.docvecs['TRAIN_NEG_' + str(i)])
    for i in range(len(test_pos)):
        test_pos_vec.append(model.docvecs['TEST_POS_' + str(i)])
    for i in range(len(test_neg)):
        test_neg_vec.append(model.docvecs['TEST_NEG_' + str(i)])
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    nb_model = BernoulliNB()
    lr_model = LogisticRegression()

    nb_model = nb_model.fit(train_pos_vec + train_neg_vec, Y)
    lr_model = lr_model.fit(train_pos_vec + train_neg_vec, Y)

    BernoulliNB(alpha = 1.0, binarize = None)
    LogisticRegression()
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    
    nb_model = GaussianNB()
    lr_model = LogisticRegression()

    nb_model.fit(train_pos_vec + train_neg_vec, Y)
    lr_model.fit(train_pos_vec + train_neg_vec, Y)

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    tp, fp, tn, fn = 0, 0, 0, 0

    for predict_value in model.predict(test_pos_vec):
        if predict_value == "pos":
            tp += 1
        else:
            fn += 1
    
    for predict_value in model.predict(test_neg_vec):
        if predict_value == "neg":
            tn += 1
        else:
            fp += 1

    accuracy = float(tp + tn)/float(tp + tn + fp + fn)

    if print_confusion:
        print("predicted:\tpos\tneg")
        print("actual:")
        print("pos\t\t%d\t%d"%(tp, fn))
        print("neg\t\t%d\t%d"%(fp, tn))
    print("accuracy: %f"%(accuracy))



if __name__ == "__main__":
    main()
