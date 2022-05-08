import numpy as np
from random import seed, random
import math


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
        p_value = np.log(self._p_count / self._i + 0.0001)
        for i in range(22):
            n_c = 0  # Counts instances where label and attribute match
            for j in range(self._i):
                # Increment n_c IF attribute and label match
                n_c += ((sample[i+1] == self._data[j][i+1]) & (sample[0] == 'p'))
            p_value += np.log((n_c + self._m * self._p) / (self._p_count + self._m))
        # Calculate comparison value for "edible" label
        e_value = np.log(self._e_count / self._i + 0.0001)
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
        self._w = np.zeros(123)
        for i in range(123):
            self._w[i] = -1 + (random() * 2)
        self._eta = 0.05  # Learning rate (hyperparameter)

    def perceptron(self, recoded):
        threshold = self._w[0]
        for i in range(122):
            threshold += self._w[i+1] * recoded[i+1]
        if threshold > 0:
            return 1
        else:
            return -1

    def train(self, example):
        recoded = recode(example)
        o = self.perceptron(recoded)
        match example[0]:
            case 'p':
                t = -1
            case _:
                t = 1
        if o != t:
            # Incorrect label. Adjust weights.
            self._w[0] += self._eta * (t - o)
            for i in range(122):
                self._w[i+1] += self._eta * (t - o) * recoded[i+1]
            return 0
        return 1

    def test(self, sample):
        recoded = recode(sample)
        o = self.perceptron(recoded)
        match sample[0]:
            case 'p':
                t = -1
            case _:
                t = 1
        return o == t


class LogisticRegression(Learner):
    def __init__(self):
        # Initialize weights to random values between 0 and 1
        seed(1)
        self._w = np.zeros(123)
        for i in range(123):
            self._w[i] = -1 + (random() * 2)
        #p = 51.8 / 48.2
        #self._w[0] = np.log(p / (1-p))
        self._eta = 0.00005  # Learning rate (hyperparameter)
        self._gradient = np.zeros(123)
        self._m = 0

    def train(self, example):
        # For diagnostic ease
        success = self.test(example)

        recoded = recode(example)
        # Standardize gradient so that it can be added to stochastically
        for i in range(123):
            self._gradient[i] *= -1 * self._m
            xi = recoded[i]
            if i == 0:
                xi = 1  # For first weight convenience
            self._gradient[i] += (recoded[0] * xi) / (1 + math.exp(recoded[0] * self._w[i] * xi))
            self._m += 1
            self._gradient[i] *= -1 / self._m
        # Update the weights
        max_weight_change = 0  # For stopping condition
        for i in range(123):
            weight_change = -1 * self._eta * self._gradient[i]
            if abs(weight_change) > max_weight_change:
                max_weight_change = abs(weight_change)
            self._w[i] += weight_change
        # Returns 0 or 1 to continue based on stopping conditions
        # return int((max_weight_change < 10**-6) | (self._m > 10000))
        # Returns success
        self._eta = 0.5 / self._m
        return success

    def test(self, sample):
        # Classifies sample based on probability (sigmoid)
        recoded = recode(sample)
        sigmoid = self._w[0]
        for i in range(122):
            sigmoid += self._w[i+1] * recoded[i+1]
        threshold = 0.5
        sigmoid = 1 / (1 + math.exp(sigmoid))
        # print("True label is " + sample[0] + ". Sigmoid assigned probability of " + str(sigmoid) + ".")
        return int(sigmoid > threshold) == recoded[0]


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
    # Cap surface
    match example[2]:
        case 'f':
            recoded[7] = 1
        case 'g':
            recoded[8] = 1
        case 'y':
            recoded[9] = 1
        case 's':
            recoded[10] = 1
    # cap-color
    match example[3]:
        case 'n':
            recoded[11] = 1
        case 'b':
            recoded[12] = 1
        case 'c':
            recoded[13] = 1
        case 'g':
            recoded[14] = 1
        case 'r':
            recoded[15] = 1
        case 'p':
            recoded[16] = 1
        case 'u':
            recoded[17] = 1
        case 'e':
            recoded[18] = 1
        case 'w':
            recoded[19] = 1
        case 'y':
            recoded[20] = 1
    # for bruises just need an if for the T or F statement
    if example[4] == 'f':
        recoded[21] = 1
    # odor
    match example[5]:
        case 'a':
            recoded[22] = 1
        case 'l':
            recoded[23] = 1
        case 'c':
            recoded[24] = 1
        case 'y':
            recoded[25] = 1
        case 'f':
            recoded[26] = 1
        case 'm':
            recoded[27] = 1
        case 'n':
            recoded[28] = 1
        case 'p':
            recoded[29] = 1
        case 's':
            recoded[30] = 1
    # gill-attachment
    match example[6]:
        case 'a':
            recoded[31] = 1
        case 'd':
            recoded[32] = 1
        case 'f':
            recoded[33] = 1
        case 'n':
            recoded[34] = 1
    # gill spacing
    match example[7]:
        case 'c':
            recoded[35] = 1
        case 'w':
            recoded[36] = 1
        case 'd':
            recoded[37] = 1
    # for gill size just need an if for the T or F statement
    if example[8] == 'n':
        recoded[38] = 1
    # gill-color
    match example[9]:
        case 'k':
            recoded[39] = 1
        case 'n':
            recoded[40] = 1
        case 'b':
            recoded[41] = 1
        case 'h':
            recoded[42] = 1
        case 'g':
            recoded[43] = 1
        case 'r':
            recoded[44] = 1
        case 'o':
            recoded[45] = 1
        case 'p':
            recoded[46] = 1
        case 'u':
            recoded[47] = 1
        case 'e':
            recoded[48] = 1
        case 'w':
            recoded[49] = 1
        case 'y':
            recoded[50] = 1
    # stalk shape
    if example[10] == 't':
        recoded[51] = 1
    # stalk root
    match example[11]:
        case 'b':
            recoded[52] = 1
        case 'c':
            recoded[53] = 1
        case 'u':
            recoded[54] = 1
        case 'e':
            recoded[55] = 1
        case 'z':
            recoded[56] = 1
        case 'r':
            recoded[57] = 1
        # SKIP MISSING VALUE FOR NOW
    # stalk surface above
    match example[12]:
        case 'f':
            recoded[59] = 1
        case 'y':
            recoded[60] = 1
        case 'k':
            recoded[61] = 1
        case 's':
            recoded[62] = 1
    # stalk surface below
    match example[13]:
        case 'f':
            recoded[63] = 1
        case 'y':
            recoded[64] = 1
        case 'k':
            recoded[65] = 1
        case 's':
            recoded[66] = 1
    # stalk color above
    match example[14]:
        case 'n':
            recoded[67] = 1
        case 'b':
            recoded[68] = 1
        case 'c':
            recoded[69] = 1
        case 'g':
            recoded[70] = 1
        case 'o':
            recoded[71] = 1
        case 'p':
            recoded[72] = 1
        case 'e':
            recoded[73] = 1
        case 'w':
            recoded[74] = 1
        case 'y':
            recoded[75] = 1
    # stalk color below
    match example[15]:
        case 'n':
            recoded[76] = 1
        case 'b':
            recoded[77] = 1
        case 'c':
            recoded[78] = 1
        case 'g':
            recoded[79] = 1
        case 'o':
            recoded[80] = 1
        case 'p':
            recoded[81] = 1
        case 'e':
            recoded[82] = 1
        case 'w':
            recoded[83] = 1
        case 'y':
            recoded[84] = 1
    # veil type
    if example[16] == 'u':
        recoded[85] = 1
    # veil color
    match example[17]:
        case 'n':
            recoded[86] = 1
        case 'o':
            recoded[87] = 1
        case 'w':
            recoded[88] = 1
        case 'y':
            recoded[89] = 1
    # ring number
    match example[18]:
        case 'n':
            recoded[90] = 1
        case 'o':
            recoded[91] = 1
        case 't':
            recoded[92] = 1
    # ring type
    match example[19]:
        case 'c':
            recoded[93] = 1
        case 'e':
            recoded[94] = 1
        case 'f':
            recoded[95] = 1
        case 'l':
            recoded[96] = 1
        case 'n':
            recoded[97] = 1
        case 'p':
            recoded[98] = 1
        case 's':
            recoded[99] = 1
        case 'z':
            recoded[100] = 1
    # spore print color
    match example[20]:
        case 'k':
            recoded[101] = 1
        case 'n':
            recoded[102] = 1
        case 'b':
            recoded[103] = 1
        case 'h':
            recoded[104] = 1
        case 'r':
            recoded[105] = 1
        case 'o':
            recoded[106] = 1
        case 'u':
            recoded[107] = 1
        case 'w':
            recoded[108] = 1
        case 'y':
            recoded[109] = 1
    # population
    match example[21]:
        case 'a':
            recoded[110] = 1
        case 'c':
            recoded[111] = 1
        case 'n':
            recoded[112] = 1
        case 's':
            recoded[113] = 1
        case 'v':
            recoded[114] = 1
        case 'y':
            recoded[115] = 1
    # habitat
    match example[22]:
        case 'g':
            recoded[116] = 1
        case 'l':
            recoded[117] = 1
        case 'm':
            recoded[118] = 1
        case 'p':
            recoded[119] = 1
        case 'u':
            recoded[120] = 1
        case 'w':
            recoded[121] = 1
        case 'd':
            recoded[122] = 1
    return recoded
