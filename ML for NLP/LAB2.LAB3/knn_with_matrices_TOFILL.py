#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse


class Examples:
    """
    a batch of examples:
    vector = vector representation of an object (dict for features with non-null values only)
    gold_class = gold class for this object
    """
    def __init__(self):
        self.gold_classes = []
        self.dict_vectors = []


class KNN:
    """
    K-NN for document classification (multiclass)

    members = 

    X_train = matrix of training example vectors
    Y_train = list of corresponding gold classes

    K = maximum number of neighbors to consider

    """
    def __init__(self, X, Y, K=1, weight_neighbors=False, verbose=False, cos_dist="similarity"):
        self.X_train = X   # (nbexamples, d)
        self.Y_train = Y   # list of corresponding gold classes


        self.cos_dist = cos_dist

        # nb neighbors to consider
        self.K = K

        # if True, the nb of neighbors will be weighted by their similarity to the example to classify
        self.weight_neighbors = weight_neighbors

        self.verbose = verbose
        
        if verbose:
            print(f"KNN classifier built with {K} neighbors")
            print(f"Weighted neighbors: {weight_neighbors}")
            print(f"Distance: {cos_dist}")

    def cosine_similarity_sorted(self, X1, X2):
        """
        Compute the cosine distance between two matrices X1 and X2
        """
        X1_norm = np.linalg.norm(X1, axis=1, keepdims=True)
        X2_norm = np.linalg.norm(X2, axis=1, keepdims=True)
        cos_dist = np.dot(X1 / X1_norm, (X2 / X2_norm).T)
        cos_dist = np.array(cos_dist)
        return cos_dist

    def calc_distance(self, X1, X2):
        """
        Compute the cosine distance between two matrices X1 and X2
        """
        X1_norm = np.linalg.norm(X1, axis=1, keepdims=True)
        X2_norm = np.linalg.norm(X2, axis=1, keepdims=True)
        cos_dist = np.dot(X1 / X1_norm, (X2 / X2_norm).T)
        cos_dist = np.array(cos_dist)
        return 1 - cos_dist

    def evaluate_on_test_set(self, X_test, Y_test, indices):
        """
        Evaluate the classifier on a test set
        """
        if self.cos_dist == "similarity":
            sorted_dist = self.cosine_similarity_sorted(X_test, self.X_train)
            print(f"sorted_dist: {sorted_dist}")
            sorted_dist_indices = np.argsort(sorted_dist, axis=1)
            print(f"sorted_dist_indices: {sorted_dist_indices}")
        elif self.cos_dist == "distance":
            sorted_dist = self.calc_distance(X_test, self.X_train)
            sorted_dist_indices = np.argsort(sorted_dist, axis=1)
        # je dois prendre les K plus proches distances, en gardant les indices
        sorted_dist_indices = sorted_dist_indices[:, -self.K:]
        # [: , -self.K:] all lines, last K columns (the K nearest neighbors)
        predictions = []
        print(f"sorted_dist_indices: {sorted_dist_indices}")
        for doc, neighbors in enumerate(sorted_dist_indices):
            neighbor_classes = {}
            if self.weight_neighbors:
                for idx in neighbors:
                    if Y_train[idx] not in neighbor_classes:
                        neighbor_classes[Y_train[idx]] = sorted_dist[doc, idx]
                    else:
                        neighbor_classes[Y_train[idx]] += sorted_dist[doc, idx]
            else:
                for idx in neighbors:
                    if Y_train[idx] not in neighbor_classes:
                        neighbor_classes[Y_train[idx]] = 1
                    else:
                        neighbor_classes[Y_train[idx]] += 1
            print(f"Doc {doc} neighbors: {neighbor_classes}")
            print(f"Neighbors_classes: {neighbor_classes}")
            predicted_class = max(neighbor_classes, key=neighbor_classes.get)
            predictions.append(predicted_class)
        accuracy = np.mean([pred == real for pred, real in zip(predictions, Y_test)])
        print(f"Accuracy: {accuracy * 100:.2f}% ({accuracy * len(Y_test)}/{len(Y_test)})")
        return accuracy


def read_examples(infile):
    """ Reads a .examples file and returns an Examples instance.
    """

    stream = open(infile)
    examples = Examples()
    dict_vector = None
    while 1:
        line = stream.readline()
        if not line:
            break
        line = line[0:-1]
        if line.startswith("EXAMPLE_NB"):
            if dict_vector != None:
                examples.dict_vectors.append(dict_vector)
            dict_vector = {}
            cols = line.split('\t')
            gold_class = cols[3]
            examples.gold_classes.append(gold_class)
        elif line:# and dict_vector != None:
            (wordform, val) = line.split('\t')
            dict_vector[wordform] = float(val)
    
    if dict_vector != None:
        examples.dict_vectors.append(dict_vector)
    return examples


def build_matrices(examples, w2i):
    nb_vocab = len(w2i)
    X=np.zeros((len(examples.dict_vectors), nb_vocab))
    Y = examples.gold_classes
    for i, example in enumerate(examples.dict_vectors):
        for word, value in example.items():
            if word in w2i:
                X[i, w2i[word]] = value
            else:
                continue
    return (X, Y)


usage = """ DOCUMENT CLASSIFIEUR using K-NN algorithm

  prog [options] TRAIN_FILE TEST_FILE

  In TRAIN_FILE and TEST_FILE , each example starts with a line such as:
EXAMPLE_NB	1	GOLD_CLASS	earn

and continue providing the non-null feature values, e.g.:
declared	0.00917431192661
stake	0.00917431192661
reserve	0.00917431192661
...

"""

parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('train_file', help='Examples\' file, used as neighbors', default=None)
parser.add_argument('test_file', help='Examples\' file, used for evaluation', default=None)
parser.add_argument("-k", '--k', default=1, type=int, help='Maximum number of nearest neighbors to consider (all values between 1 and K will be tested). Default=1')
parser.add_argument('-v', '--verbose',action="store_true",default=False,help="If set, triggers a verbose mode. Default=False")
parser.add_argument('-w', '--weight_neighbors', action="store_true", default=False,help="If set, neighbors will be weighted when scoring classes. Default=False")
parser.add_argument('-c', '--cos_dist', choices=['similarity', 'distance'], default="similarity", help='Choose between cosine similarity or cosine distance. Default=similarity')
args = parser.parse_args()




#------------------------------------------------------------
# Reading training and test examples :

train_examples = read_examples(args.train_file) # .dev.examples
test_examples = read_examples(args.test_file) # .test.examples
cos_dist = args.cos_dist # similarity or distance

print(f"Read {len(train_examples.dict_vectors)} trained (.dev) examples") # On va baser notre KNN sur ces exemples
print(f"Read {len(test_examples.dict_vectors)} test (.test) examples") # On va évaluer notre KNN sur ces exemples

# ------------------------------------------------------------
# Building indices for vocabulary in TRAINING examples

w2i = {}
i2w = {}
set_words = set()
for example in train_examples.dict_vectors:
    for word in example.keys():
        set_words.add(word)
for index, word in enumerate(set_words):
    w2i[word] = index
    i2w[index] = word

# ------------------------------------------------------------
# Organize the data into two matrices for document vectors
#                   and two lists for the gold classes
(X_train, Y_train) = build_matrices(train_examples, w2i)
(X_test, Y_test) = build_matrices(test_examples, w2i)
print(f"Trained (.dev) matrix has shape {X_train.shape}")
print(f" Testing (.test) matrix has shape {X_test.shape}")
myclassifier = KNN(X=X_train,
                   Y=Y_train,
                   K=args.k,
                   weight_neighbors=args.weight_neighbors,
                   verbose=args.verbose)

print("Evaluating on test...")
for k in range(1, args.k + 1):
    print(f"K = {k}")
    myclassifier = KNN(X=X_train,
                   Y=Y_train,
                   K=k,
                   weight_neighbors=args.weight_neighbors,
                   verbose=args.verbose,
                   cos_dist=cos_dist)
    accuracies = myclassifier.evaluate_on_test_set(X_test, Y_test, i2w)