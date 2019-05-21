from random import seed
from random import randrange
from math import sqrt
from csv import reader

# wczytanie danych z pliku


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# konwersja kolumn klas wejsciowych na typ zmiennoprzecinkowy


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# konwersja kolumny klasy wyjsciowej na typ calkowity


def str_column_to_integer(dataset, column):
    for row in dataset:
        row[column] = int(row[column].strip())

# podzial zbioru na k podzbiorow


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

# obliczanie poprawnosci klasyfikacji k-tego podzbioru


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# wykonanie algorytmu z wykorzystaniem walidacji krzyzowej


def evaluate_algorithm(folds, algorithm, *args):
    scores = list()
    for fold in folds:
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

# obliczanie normy Euklidesowej miedzy dwoma wektorami


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# poszukiwanie najblizszego wektora wedlug odleglosci od niego


def get_best_matching_unit(codebooks, test_row):
    distances = list()
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]

# przewidywanie najbardziej pasujacego wektora
# z uzyciem zbioru wektorow uczacych


def predict(codebooks, test_row):
    bmu = get_best_matching_unit(codebooks, test_row)
    return bmu[-1]

# wygenerowanie losowego wektora uczacego


def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(n_records)][i]
                for i in range(n_features)]
    return codebook

# uczenie zbioru wektorow uczacych


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
    return codebooks

# algorytm LVQ


def learning_vector_quantization(train, test,
                                 n_codebooks, lrate, epochs):
    codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
    predictions = list()
    for row in test:
        output = predict(codebooks, row)
        predictions.append(output)
    return(predictions)


# przeprowadzanie eksperymentow
seed(1)
# wczytanie i przygotowanie danych
filename = 'data.csv'
dataset = load_csv(filename)

for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
str_column_to_integer(dataset, len(dataset[0])-1)

# podzial danych na 10 podzbiorow
dataset_split = cross_validation_split(dataset, 10)


# otworzenie plikow, w ktorych beda zapisywane
# aktualny stan petli i wyniki
f4 = open('output1.txt', 'w')
f5 = open('output2.txt', 'w')
f6 = open('output3.txt', 'w')
f7 = open('output4.txt', 'w')

# utworzenie testowanego zakresu wspolczynnika uczenia,
# liczby neuronow oraz liczby epok
learn_rate_list = list()
for i in range(9):
    learn_rate_list.append(float((i+1)/100))

for i in range(10):
    learn_rate_list.append(float((i+1)/10))

learn_rate_list.append(0.99)
learn_rate_list.sort()

n_codebooks = list()
for i in range(10):
    n_codebooks.append((i * 10) + 10)

n_epochs = [10, 50, 100]

# petla przeprowadzajaca eksperymenty
for learn_rate in learn_rate_list:
    for n_epoch in n_epochs:
        for n_codebook in n_codebooks:
            scores = evaluate_algorithm(
                dataset_split, learning_vector_quantization,
                n_codebook, learn_rate, n_epoch)
            print('PARAMETERS: LR %.2f, EPOCH %d, NEURONS %d' % (
                learn_rate, n_epoch, n_codebook))
            result = sum(scores)/float(len(scores))
            print('Mean Accuracy: %.2f%%' % result)
            f4.write('%.2f\n' % learn_rate)
            f5.write('%.2f\n' % n_epoch)
            f6.write('%d\n' % n_codebook)
            f7.write('%.2f\n' % result)
f4.close()
f5.close()
f6.close()
f7.close()
