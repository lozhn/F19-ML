import numpy as np
import pandas as pd
import pydot
# code is based on this resource - http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html


def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}


def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def information_gain(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


def information_gain_ratio(y, x):
    IG = information_gain(y, x)
    IV = entropy(x)
    return IG / IV if IV != 0 else 0


def is_pure(s):
    return len(set(s)) == 1


def recursive_split(x, y, fields, max_depth, current_depth=0):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0 or current_depth >= max_depth:
        (values, counts) = np.unique(y, return_counts=True)
        ind = np.argmax(counts)
        return y[ind]

    # We get attribute that gives the highest information gain
    gain = np.array([information_gain_ratio(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)
    print('Max value for attribute {} = {}'.format(
        fields[selected_attr], gain[selected_attr]))
    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        (values, counts) = np.unique(y, return_counts=True)
        ind = np.argmax(counts)
        return y[ind]

    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}

    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["{}".format(k)] = recursive_split(
            x_subset, y_subset, fields, max_depth, current_depth + 1)
    final_res = {str(selected_attr) + '_' + fields[selected_attr]: res}
    return final_res


def predict(tree, x):
    if isinstance(tree, dict):
        for k, v in tree.items():
            if isinstance(v, dict):
                idx = int(k.split('_')[0])
                for k_, v_ in v.items():
                    if str(x[idx]) == str(k_):
                        return predict(v_, x)
            else:
                return v

    else:
        return tree


def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name, color='green')
    graph.add_edge(edge)


def visit(node, parent=None):
    for k, v in node.items():
        if isinstance(v, dict):
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            draw(k, v)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# train_set = pd.read_csv(
#     'titanic_modified.csv')
train_set = pd.read_csv('tennis.csv')


X = train_set.iloc[:, :4].values
y = train_set.iloc[:, 4].values

fields = list(train_set.columns.values)
max_depth = 4
tree = recursive_split(X, y, fields, max_depth)
print(tree)

X_test = X[3]
y_pred = predict(tree, X_test)
print('prediction for {} is {}'.format(X_test, y_pred))
graph = pydot.Dot(graph_type='graph')
visit(tree)
graph.write_png('graph.png')
