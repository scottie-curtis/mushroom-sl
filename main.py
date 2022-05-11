import numpy as np
from learners import NaiveBayes, Perceptron, LogisticRegression, DecisionTree
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filename = "agaricus-lepiota.data"
    data = np.genfromtxt(filename, dtype='str', delimiter=",")
    num_instances = len(data)               # Rows in dataset; initially, 8124
    num_train = int(num_instances * 0.75)   # Using initial values, 6093
    num_test = num_instances - num_train    # '', 2031

    # Initialize learners
    nb = NaiveBayes()
    p = Perceptron()
    lr = LogisticRegression()
    dt = DecisionTree()

    # Initialize diagnostics
    examples = []
    p_success_rate = []
    nb_success_rate = []
    lr_success_rate = []
    dt_success_rate = []

    # Graphing variables
    fig = plt.figure()
    plt.title("Logistic Regression Success Rate")
    plt.xlabel("Training Examples")
    plt.ylabel("Success Rate Over Test Set")
    plt.xlim([1, num_train])
    plt.ylim([0, 1.1])

    # Below are two different methods for testing models. The unused one should be block-commented out.
    # Program is also optimized for one model at a time. Replace (nb/p/lr/dt) in variable names to switch models.

    '''# Conventional train-test split. Tests entire test set with each new training example.
    # Best results, but tremendously slow.
    for i in range(num_train):
        p_successes = 0
        DecisionTree.train(dt, data[i])
        for j in range(num_test):
            p_successes += DecisionTree.test(dt, data[num_train + j])
        examples.append(i + 1)
        p_success_rate.append(p_successes / num_test)
        print("Trained " + str(i+1) + " examples, " + str(p_successes)
              + "/" + str(num_test) + " testing examples classified correctly.")'''


    # Progressive testing. Tests every sample and then immediately uses it to train model.
    # Not as informative, since it doesn't reveal overfitting, but MUCH faster.
    p_successes = 0
    for i in range(40):
        p_successes += DecisionTree.train(dt, data[i])
        examples.append(i + 1)
        p_success_rate.append(p_successes / (i + 1))
        print("Trained " + str(i+1) + " examples, " + str(p_successes)
              + "/" + str(i+1) + " training examples classified correctly.")
    dt.print()

    plt.plot(examples, p_success_rate)
    plt.show()

