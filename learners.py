import numpy as np
from random import seed, random

class Learner:
    def train(self, example: list):
        pass

    def test(self, sample: list):
        pass


class NaiveBayes(Learner):
    def __init__(self):
        self._data = np.empty((6193, 23), dtype=str)     # Stores training data
        self._i = 0     # Iterator
        self._p_count = 0   # Number of poisonous examples
        self._e_count = 0   # Number of edible examples
        self._m = 1         # Equivalent sample size (hyperparameter)
        self._p = 1 / 22    # Prior estimate of probability

    def train(self, example):
        for i in range(len(example)):
            self._data[self._i][i] = example[i]
        self._p_count += (example[0] == 'p')  # Increment poisonous counter
        self._e_count += (example[0] == 'e')  # Increment edible counter
        self._i += 1    # Increment general counter

    def test(self, sample):
        # Calculate comparison value for "poisonous" label
        p_value = np.log(self._p_count / self._i)
        for i in range(22):
            n_c = 0  # Counts instances where label and attribute match
            for j in range(self._i):
                # Increment n_c IF attribute and label match
                n_c += ((sample[i+1] == self._data[j][i+1]) & (sample[0] == 'p'))
            p_value += np.log((n_c + self._m * self._p) / (self._p_count + self._m))
        # Calculate comparison value for "edible" label
        e_value = np.log(self._e_count / self._i)
        for i in range(22):
            n_c = 0  # Counts instances where label and attribute match
            for j in range(self._i):
                # Increment n_c IF attribute and label match
                n_c += ((sample[i + 1] == self._data[j][i + 1]) & (sample[0] == 'e'))
            e_value += np.log((n_c + self._m * self._p) / (self._e_count + self._m))
        if e_value > p_value:
            label = 'e'
        else:
            label = 'p'
        return int(label == sample[0])  # Returns 1 if label correct, 0 if incorrect





class Perceptron(Learner):
    def __init__(self):
        # Initialize weights to random values between 0 and 1
        seed(1)
        self._w = np.zeros(23)
        for i in range(23):
            self._w[i] = -1 + (random() * 2)

    def perceptron(self, sample):
        threshold = self._w[0]
        for i in range(22):
            threshold += self._w[i+1] *

    def train(self, example):



class LogisticRegression(Learner):
    pass  # TODO: fill


class DecisionTree(Learner):
    pass  # TODO: fill

# Recodes categorical values into dummy values to be used in regression.
# Details are not necessarily important to understand.
def recode(example):
    recoded = np.zeros(123)
    # If there are only 2 values, just use an if statement.
    # For consistency, first one is 0, second is 1.
    if example[0] == 'e':
        recoded[0] = 1
    # For any >2, use a number of columns equal to the number of options.
    match example[1]:
        case 'b':
            recoded[1] = 1
        case 'c':
            recoded[2] = 1
        case 'x':
            recoded[3] = 1
        case 'f':
            recoded[4] = 1
        case 'k':
            recoded[5] = 1
        case 's':
            recoded[6] = 1
    match example[2]:
        # TODO: etc
