import numpy as np
from learners import NaiveBayes, Perceptron, LogisticRegression, DecisionTree
import matplotlib.pyplot as plt

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

    # Initialize diagnostics
    examples = []
    nb_successes = 0
    p_successes = 0
    p_success_rate = []
    nb_success_rate = []

    # Graphing variables
    fig = plt.figure()
    plt.title("Perceptron Success Rate")
    plt.xlabel("Training Examples")
    plt.ylabel("Success Rate over Test Set")
    plt.xlim([1, num_instances])
    plt.ylim([0, 1.1])

    # Train learners
    for i in range(num_train):
        example = data[i]
        Perceptron.train(p, example)
        for j in range(num_test):
            sample = data[num_train + j]
            # nb_successes += NaiveBayes.test(nb, sample)
            p_successes += Perceptron.test(p, sample)
        # p_successes += Perceptron.train(p, example)
        p_success_rate.append(float(p_successes / num_test / (i + 1)))
        examples.append(i + 1)
        print(i)
        '''
    for i in range(num_test):
        sample = data[num_train + i]
        print(i)
        nb_successes += NaiveBayes.test(nb, sample)
        # p_successes += Perceptron.test(p, sample)
        print(num_train + i)
    nb_success_rate.append(float(nb_successes / num_test))
    examples.append(10)
    for i in range(num_train - 10):
        example = data[i + 10]
        NaiveBayes.train(nb, example)

    # Test learners
    for i in range(num_test):
        sample = data[num_train + i]
        print(i)
        nb_successes += NaiveBayes.test(nb, sample)
        # p_successes += Perceptron.test(p, sample)
        print(num_train + i)

    nb_success_rate.append(1.0)
    examples.append(num_train)

    # Print results
    print("Naive Bayes Accuracy: " + str(nb_successes / num_test))
    print("Perceptron Accuracy: " + str(p_successes / num_test))
    '''

    plt.plot(examples, p_success_rate)
    plt.show()
