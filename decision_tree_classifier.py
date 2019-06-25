"""
Here I will make a decision-tree-algorithm which will trained by a set of training data,
and then used to classify a set of test data. The algorithm adopts a greedy divide-and-conquer
strategy: always test the most important attribute/feature first. Most important means making
the most difference to the classification.

"""

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # To ignore some future warnings from pandas
import math
import random

def get_datasets():
    try:
        # Converting the txt.-datafiles to pandas as it is nice to work with
        train = pd.read_csv('./Dataset/training.txt', sep="	", header=None)
        test = pd.read_csv('./Dataset/test.txt', sep= "	", header=None)
        features = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "y"]
        train.columns = features
        test.columns = features
        return train, test
    except:
        print("Error when load datasets")

def plurality_value(parent_examples):
    # Chooses random if tie, else the most common value for 'y'
    # (value_counts() is sorted, therefore index = 0)
    r = 0
    if parent_examples['y'].value_counts().index[0] == parent_examples['y'].value_counts().index[1]:
        r = random.randint(1)
    return parent_examples['y'].value_counts().index[r]

def allValuesEqual(examples):
    return len(examples['y'].value_counts()) == 1


def partition(examples, column, value):
    """  
    :param examples: input dataset/rows 
    :param column: column to match 
    :param value: condition value to match each row with, equals 1 or 2
    :return: matching rows and nonmatching rows
    """

    true_rows, false_rows = examples.copy(), examples.copy()
    for index, row in examples.iterrows():
        if row[column] == value:
            false_rows = false_rows.drop([index])
        else:
            true_rows = true_rows.drop([index])
    return true_rows, false_rows


def get_entropy(examples):
    # Calculating the number of the examples that have ouput (y) == 1
    number_of_ones = 0
    for index, row in examples.iterrows():
        if row['y'] == 1:
            number_of_ones += 1

    q = number_of_ones/len(examples) if len(examples) != 0 else 0
    return - ( q*math.log2(q) + (1-q)*math.log2((1-q)) ) if (q != 0 and q!= 1) else 0


def get_remainder(one_rows, two_rows, entropy1, entropy2):
    total = len(one_rows) + len(two_rows)
    p, n = len(one_rows), len(two_rows)
    return (p/total)*entropy1 + (n/total)*entropy2


def find_best_split(examples, attributes, current_entropy):
    """
    :param examples: examples to consider
    :param attributes: attributes to consider
    :param current_entropy: the entropy we are comparing the different new entropies against
    :return: the attributes that makes a split which results in the most information gain
    """

    # For each attribute, calculate info gain and choose attribute with highest info gain
    best_gain = 0
    best_attribute = None
    info_gains = []
    for attribute in attributes:
        # Partitions the examples based on whether or not their value for the attribute equals 1
        ones, twoes = partition(examples, attribute, 1)
        # print(ones.shape

        # Calculating entropies for the two partitions
        entropy1, entropy2 = get_entropy(ones), get_entropy(twoes)

        # Calculating the remainder using the entropies
        remainder = get_remainder(ones, twoes, entropy1, entropy2)

        # Skip this split if it doesn't divide the dataset
        if len(ones) == 0 and len(twoes) == 0:
            continue

        # Information gain
        info_gain = current_entropy - remainder
        info_gains.append(info_gain)
        if info_gain > best_gain:
            best_gain, best_attribute = info_gain, attribute
    return best_attribute if best_gain != 0 else random.choice(attributes)


def decision_tree_learning(examples, attributes, parent_examples):
    """
    :param examples: examples to consider in this iteration
    :param attributes: attributes 'available' in this iteration, meaning not previously used in the path from 
    root to this node 
    :param parent_examples: the examples as they are before the split 
    :return: a complete Decision Tree (of class Tree) 
    """

    # If examples are empty return most common output value among the parent examples (before the last split)
    if len(examples) == 0:
        return plurality_value(parent_examples)

    # If all examples have the same output value, then the partition is pure and we return the classification
    elif allValuesEqual(examples):
        #print(examples.iloc[0]['y'])
        return examples.iloc[0]['y']

    # If attributes are empty (no more partition possible) return the plurality value of current examples
    elif len(attributes) == 0:
        return plurality_value(examples)


    # Else we continue the partition

    current_entropy = get_entropy(examples)

    best_attribute= find_best_split(examples, attributes, current_entropy) # Importance function version 1
    #best_attribute = random.choice(attributes) # Importance function version 2

    tree = Tree(best_attribute)
    # Making a copy and then removing the attribute from that copy,
    # as we need the attribute available in other nonsuccessor branches (when the recursion "comes back" again)
    attr_copy = attributes[:]
    attr_copy.remove(best_attribute)

    for value in [1, 2]:
        next_examples = examples.loc[examples[best_attribute] == value]
        subtree = decision_tree_learning(next_examples, attr_copy, examples)
        tree.add_branch(value, subtree)
    return tree



class Tree:
    def __init__(self, root, branches=None):
        self.root = root
        if branches == None:
            branches = {}
        self.branches = branches

    def add_branch(self, label, branch):
        self.branches[label] = branch


def predict(tree, example):
    value = example[tree.root]
    if value == 1:
        if isinstance(tree.branches[1], Tree):
            return predict(tree.branches[1], example)
        else:
            return tree.branches[1]
    else:
        if isinstance(tree.branches[2], Tree):
            return predict(tree.branches[2], example)
        else:
            return tree.branches[2]


def print_tree(tree, value="", level=0):
    print("\t" * level + str(value), end = " -> ")
    attribute = tree.root if isinstance(tree, Tree) else str(tree)
    print(attribute, end = "\n")

    if isinstance(tree, Tree):
        for label, subtree in tree.branches.items():
            print_tree(subtree, label, level + 1)

def main():


    # Load data
    train, test = get_datasets()
    train_x = train.drop(columns=['y'])
    train_y = train['y']
    print(train_x.head())
    test_x = test.drop(columns=['y'])
    test_y = test['y']
    print("--------------------------------------------")

    # Heatmap of correlations between attributes and 'y' (Class attribute)
    import seaborn as sns
    import matplotlib.pyplot as plt
    hm = sns.heatmap(train.corr(), annot=True, linewidth=.5, cmap='Blues')
    hm.set_title(label='Heatmap of correlations', fontsize=20)
    plt.show()

    # Build tree
    tree = decision_tree_learning(train, ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven'], train)

    # Visualize tree
    print_tree(tree)

    # Check accuracy
    from sklearn.metrics import accuracy_score
    predictions = []
    for index, example in test_x.iterrows():
        prediction = predict(tree, example)
        predictions.append(prediction)
    acc = accuracy_score(test_y, predictions)
    print("Accuracy with my greedy Importance Decision Tree: ", acc)


    # Comparison with different classifiers from sklearn

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    sk_tree = DecisionTreeClassifier()
    sk_tree.fit(train_x, train_y)
    predictions = sk_tree.predict(test_x)
    print("Accuracy with sklearn Decision Tree: ", accuracy_score(test_y, predictions))

    sk_tree = RandomForestClassifier()
    sk_tree.fit(train_x, train_y)
    predictions = sk_tree.predict(test_x)
    print("Accuracy with sklearn Random Forest: ", accuracy_score(test_y, predictions))

    sk_tree = KNeighborsClassifier()
    sk_tree.fit(train_x, train_y)
    predictions = sk_tree.predict(test_x)
    print("Accuracy with sklearn K Nearest Neighbors Classifier: ", accuracy_score(test_y, predictions))


main()















