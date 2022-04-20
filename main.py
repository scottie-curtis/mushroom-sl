import numpy as np
from learners import NaiveBayes, Perceptron, LogisticRegression, DecisionTree

if __name__ == "__main__":
    filename = "agaricus-lepiota.data"
    data = np.genfromtxt(filename, dtype='str', delimiter=",")
    num_instances = len(data)               # Rows in dataset; initially, 8124
    num_train = int(num_instances * 0.75)   # Using initial values, 6193
    num_test = num_instances - num_train    # '', 2031

    # Initialize learners
    nb = NaiveBayes()
    p = Perceptron()
    lr = LogisticRegression()
    dt = DecisionTree()

    # Train learners
    for i in range(num_train):
        example = data[i]
        NaiveBayes.train(nb, example)

    # Test learners
    nb_successes = 0
    for i in range(num_test):
        sample = data[num_train + i]
        print(i)
        nb_successes += NaiveBayes.test(nb, sample)

    # Print results
    print("Naive Bayes Accuracy: " + str(nb_successes / num_test))
