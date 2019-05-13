# LVQ for the Ionosphere Dataset
from random import seed
from random import randrange
from math import sqrt
from csv import reader
import time


# Load a CSV file


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer


def str_column_to_integer(dataset, column):
    for row in dataset:
        row[column] = int(row[column].strip())

# Split a dataset into k folds


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split


def evaluate_algorithm(folds, algorithm, *args):
    # folds = cross_validation_split(dataset, n_folds)
    scores = list()
    iterator = 1
    for fold in folds:
        #print('Fold: %d' % iterator)
        iterator += 1
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# calculate the Euclidean distance between two vectors


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the best matching unit


def get_best_matching_unit(codebooks, test_row):
    distances = list()
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]

# Make a prediction with codebook vectors


def predict(codebooks, test_row):
    bmu = get_best_matching_unit(codebooks, test_row)
    return bmu[-1]

# Create a random codebook vector


def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(n_records)][i] for i in range(n_features)]
    # print(codebook)
    return codebook

# Train a set of codebook vectors


def train_codebooks(train, n_codebooks, lrate, epochs):
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    for epoch in range(epochs):
        rate = lrate * (1.0-(epoch/float(epochs)))
        for row in train:
            bmu = get_best_matching_unit(codebooks, row)
            for i in range(len(row)-1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
        # if epoch % 100 == 0:
            #print('Epoch: %d' % epoch)
    return codebooks

# LVQ Algorithm


def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
    codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
    predictions = list()
    for row in test:
        output = predict(codebooks, row)
        predictions.append(output)
    return(predictions)


# Test LVQ on Ionosphere dataset
seed(1)
# load and prepare data
filename = 'data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_integer(dataset, len(dataset[0])-1)

dataset_split = cross_validation_split(dataset, 10)

# evaluate algorithm

f4 = open('output10.txt', 'w')
f5 = open('output11.txt', 'w')
f6 = open('output12.txt', 'w')

how = 40

#learn_rate = 0.03
n_epochs = 10
#n_codebooks = 100
for learn_rate in range(how):
    # for n_epochs in range(100, 500, 100):
    for n_codebooks in range(how):
        starttime = time.time()
        scores = evaluate_algorithm(
            dataset_split, learning_vector_quantization, int(n_codebooks * (100/how) + (100/how)), float((learn_rate + 1)/how), n_epochs)
        elapsedtime = time.time() - starttime
        print('PARAMETERS: LR %.2f, EPOCH %d, NEURONS %d, TIME [s]: %d' %
              (float((learn_rate + 1)/how), n_epochs, int(n_codebooks * (100/how) + (100/how)), elapsedtime))
        #print('Scores: %s' % scores)
        f4.write('%.2f\n' % float((learn_rate + 1)/how))
        f5.write('%d\n' % int(n_codebooks * (100/how) + (100/how)))
        result = sum(scores)/float(len(scores))
        print('Mean Accuracy: %.2f%%' % result)
        f6.write('%.2f\n' % result)
# starttime = time.time()
# scores = evaluate_algorithm(
#     dataset_split, learning_vector_quantization, n_codebooks, learn_rate, n_epochs)
# elapsedtime = time.time() - starttime
# print('Params: LR %.2f%% EP %d NV %d time: %d' %
#       (learn_rate, n_epochs, n_codebooks, elapsedtime))
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


# dopracowac wybor danych do podzbioru
# lrate = [0.01 0.1:0.1:0.9 0.95 0.99 1.0]
# matlab 6.5.1 r2003sp1
# 1. ilosc epok
# 2. ilosc neuronow
# 3. learning rate


# f = open('output3.txt', 'w')
# for i in range(100):
#     w = randrange(3) + 1
#     f.write('%d\n' % w)
# f.close()
f4.close()
f5.close()
f6.close()
