import pandas as pd
from math import log2
from collections import namedtuple

Node = namedtuple('Node', ['attribute', 'branches'])

def entropy(pos, neg):
    total = pos + neg
    if pos == 0 or neg == 0:
        return 0
    return -sum(p / total * log2(p / total) for p in [pos, neg])

def gini_index(values):
    return 1 - sum(val ** 2 for val in values)

def calc_gini(df, attribute, label_col):
    total_count = len(df)
    gini = 0
    for value in df[attribute].unique():
        subset = df[df[attribute] == value]
        subset_count = len(subset)
        subset_probs = subset[label_col].value_counts() / subset_count
        gini += subset_count / total_count * gini_index(subset_probs)
    return gini

def id3(df, label_col):
    if len(df[label_col].unique()) == 1:
        return df[label_col].iloc[0]
    
    gains = {}
    total_entropy = entropy(*df[label_col].value_counts())
    for attribute in df.columns.difference([label_col]):
        avg_entropy = 0
        for value in df[attribute].unique():
            subset = df[df[attribute] == value]
            subset_entropy = entropy(*subset[label_col].value_counts())
            avg_entropy += len(subset) / len(df) * subset_entropy
        gains[attribute] = total_entropy - avg_entropy
    
    best_attr = max(gains, key=gains.get)
    branches = {}
    for value in df[best_attr].unique():
        subset = df[df[best_attr] == value].drop(columns=[best_attr])
        branches[value] = id3(subset, label_col)
    
    return Node(best_attr, branches)

def print_tree(tree, level=0, branch=None):
    indent = "  " * level
    if isinstance(tree, Node):
        print(f"{indent}{branch} -> {tree.attribute}")
        for branch, subtree in tree.branches.items():
            print_tree(subtree, level + 1, branch)
    else:
        print(f"{indent}{branch} -> {tree}")

if __name__ == "__main__":
    df = pd.read_csv('ass2data.csv')
    label_col = 'Buys_Computer'
    
    decision_tree = id3(df, label_col)
    print_tree(decision_tree)
