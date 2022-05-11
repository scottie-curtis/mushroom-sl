import numpy as np
from random import seed, random
import math


# Various print() statements are present in this code. Uncomment to see progress for diagnostic purposes.
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
        success = 0
        if self._i != 0:
            success = self.test(example)
        for i in range(len(example)):
            self._data[self._i][i] = example[i]
        self._p_count += (example[0] == 'p')  # Increment poisonous counter
        self._e_count += (example[0] == 'e')  # Increment edible counter
        self._i += 1    # Increment general counter
        return success

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
        # p = 51.8 / 48.2 # Alternate ideas for starting weights, scrapped
        # self._w[0] = np.log(p / (1-p))
        self._eta = 0.2  # Learning rate (hyperparameter)
        self._gradient = np.zeros(123)
        self._m = 0

    def train(self, example):
        # For diagnostic ease
        success = self.test(example)

        recoded = recode(example)

        w_xi = self._w[0]
        for i in range(122):
            w_xi += self._w[i+1] * recoded[i+1]
        # Standardize gradient so that it can be added to stochastically
        for i in range(123):
            self._gradient[i] *= -1 * self._m
            xi = recoded[i]
            if i == 0:
                xi = 1  # For first weight convenience
            yi = recoded[0]
            if yi == 0:
                yi = -1  # Calculation only works if values are -1 and 1
            self._gradient[i] += (yi * xi) / (1 + math.exp(yi * w_xi))
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
    def __init__(self):
        self._data = np.empty((6193, 123))  # Stores training data
        self._i = 0  # Counter to store data correctly
        self._root = Node("-")  # This will be overwritten by the first example trained

    def train(self, example):
        # print("Training decision tree...")
        # Diagnostic: test training example
        correct = self.test(example)
        # Read in training example, recoded
        # Recoding would not be necessary for many-branched trees, but this implementation will be a binary tree
        self._data[self._i] = recode(example)
        self._i += 1
        # Construct tree
        # print("Constructing tree.")
        indices = np.zeros(123)
        for i in range(123):  # Generates an array of indices to label nodes
            indices[i] = int(i)
        # print("Time to create a tree!")
        self._root = id3(self._data, self._i, indices, 0)
        return correct

    def test(self, sample):
        # print("Testing sample...")
        recoded = recode(sample)
        check = self._root
        hypothesis = 0  # Will never be returned, but makes the IDE happy
        while not (str(check.data) == "-" or str(check.data) == "+"):
            attribute = int(check.data)
            # print("Decision attribute: " + str(attribute))
            if recoded[attribute] == 0:
                check = check.left
                # print("This attribute is 0. Proceed down the left branch.")
            else:
                check = check.right
                # print("This attribute is 1. Proceed down the right branch.")
        if str(check.data) == "-":
            hypothesis = 0
            # print("We will hypothesize that this sample is poisonous. True label: " + sample[0])
        if str(check.data) == "+":
            hypothesis = 1
            # print("We will hypothesize that this sample is edible. True label: " + sample[0])
        # print("Test complete. Decision tree said it was " + str(hypothesis) + " and it was actually "
        #       + str(recoded[0]) + ".")
        return int(hypothesis == recoded[0])

    def print(self):
        self._root.print_tree()


# Function that takes data and returns a decision tree
def id3(data, size, indices, depth):
    # Check for all-positive or all-negative data
    p_total = 0
    e_total = 0
    # print("Checking through all " + str(size) + " stored training examples!")
    for i in range(size):
        if data[i][0] == 0:
            p_total += 1
        else:
            e_total += 1
    if e_total == 0:
        # print("All examples on this branch are poisonous.")
        return Node("-")
    if p_total == 0:
        # print("All examples on this branch are edible.")
        return Node("+")
    attributes = len(data[0]) - 1  # Attributes that are not the target
    # print("Checking " + str(attributes) + " attributes.")
    # Check for attributes empty, AND make sure the tree isn't getting too complex.
    if attributes == 0 or depth == 6:
        # Return a node with the most common label
        if p_total > e_total:
            # print("Most examples I see are poisonous.")
            return Node("-")
        else:
            # print("Most examples I see are edible.")
            return Node("+")
    max_info_gain = -65535  # Keeps track of highest information gain
    max_info_gain_index = 0  # Keeps track of the index with the highest information gain
    for i in range(attributes):
        # print("In the attribute loop, index " + str(i))
        p_left = 0  # Num examples with poisonous label and 0 for the relevant attribute
        p_right = 0  # Num examples with poisonous label and 1 for the relevant attribute
        e_left = 0  # Num examples with edible label and 0 for the relevant attribute
        e_right = 0  # Num examples with edible label and 1 for the relevant attribute
        # Increment these values for entropy calculation
        for j in range(size):
            if data[j][i+1] == 0:
                if data[j][0] == 0:
                    p_left += 1
                else:
                    e_left += 1
            else:  # 1
                if data[j][0] == 0:
                    p_right += 1
                else:
                    e_right += 1
        # Calculate entropy of the full set
        # First check for p_i == 0, to avoid log errors
        if p_left + p_right == 0:
            p_log_p = 0
        else:
            p_log_p = (p_left + p_right) / size * math.log((p_left + p_right) / size, 2)
        if e_left + e_right == 0:
            e_log_e = 0
        else:
            e_log_e = (e_left + e_right) / size * math.log((e_left + e_right) / size, 2)
        entropy = -1 * p_log_p - e_log_e
        # Calculate entropies of new branches
        # First check for p_i == 0, to avoid log errors
        if p_left == 0:
            p_log_p = 0
        else:
            p_log_p = p_left / (p_left + e_left) * math.log(p_left / (p_left + e_left), 2)
        if e_left == 0:
            e_log_e = 0
        else:
            e_log_e = e_left / (p_left + e_left) * math.log(e_left / (p_left + e_left), 2)
        entropy_left = -1 * p_log_p - e_log_e
        if p_right == 0:
            p_log_p = 0
        else:
            p_log_p = p_right / (p_right + e_right) * math.log(p_right / (p_right + e_right), 2)
        if e_right == 0:
            e_log_e = 0
        else:
            e_log_e = e_right / (p_right + e_right) * math.log(e_right / (p_right + e_right), 2)
        entropy_right = -1 * p_log_p - e_log_e
        info_gain = entropy - (p_left + e_left) / size * entropy_left - \
            (p_right + e_right) / size * entropy_right
        if info_gain > max_info_gain:
            # print("This attribute (" + str(i+1) + ") works better. We'll use it.")
            max_info_gain_index = i + 1
            max_info_gain = info_gain
    # Set root node to the attribute that best classifies examples
    root = Node(indices[max_info_gain_index])
    # print("Attribute selected to be a node: " + str(indices[max_info_gain_index]))
    # Divide data into left and right branches
    data_left = np.copy(data)
    size_left = size
    size_right = size
    for i in range(size):  # Delete rows
        if data_left[i][max_info_gain_index] == 1:
            data_left = np.delete(data_left, obj=i, axis=0)
            i -= 1  # Set index back one (i won't change next iteration, since we deleted a row)
            size_left -= 1
    data_left = np.delete(data_left, obj=max_info_gain_index, axis=1)  # Delete relevant column
    data_right = np.copy(data)
    for i in range(size):  # Delete rows
        if data_right[i][max_info_gain_index] == 0:
            data_right = np.delete(data_right, obj=i, axis=0)
            i -= 1  # Set index back one (i won't change next iteration, since we deleted a row)
            size_right -= 1
    data_right = np.delete(data_right, obj=max_info_gain_index, axis=1)  # Delete relevant column
    indices = np.delete(indices, obj=max_info_gain_index)
    # Recursively create left and right branches
    root.left = id3(data_left, size_left, indices, depth + 1)
    root.right = id3(data_right, size_right, indices, depth + 1)
    # Return the root node
    return root


# Used in the construction of decision trees
class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def print_tree(self):
        print("Node data: " + str(self.data))
        if self.left is not None:
            print("Left child: ")
            self.left.print_tree()
        if self.right is not None:
            print("Right child: ")
            self.right.print_tree()


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
