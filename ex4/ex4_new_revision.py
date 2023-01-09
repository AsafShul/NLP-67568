import nltk
# nltk.download('dependency_treebank') # uncomment to download the module
from nltk.corpus import dependency_treebank

import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")

import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split
from collections import namedtuple, defaultdict, Counter

from networkx import DiGraph
from networkx.algorithms import maximum_spanning_arborescence

# constants:
WORD = 'word'
POS = 'tag'
TAILS = 'deps'
ID = 0

ARC = namedtuple('ARC', ['head', 'tail', 'weight'])


def max_spanning_arborescence_nx(arcs):
    """
    Wrapper for the networkX min_spanning_tree to follow the original API
    :param arcs: list of Arc tuples
    We assume that 0 is the only possible root over the set of edges given to the algorithm.
    """
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = maximum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result


def _build_dict(corpus, category):
    """
    create an index dict
    """
    unique_words = []
    [unique_words.extend([d[category] for d in tree.nodes.values() if d[category] is not None]) for tree in corpus]
    unique_words = list(np.unique(unique_words))
    if category == WORD:
        unique_words = [None] + unique_words
    return Counter(dict(zip(unique_words, np.arange(len(unique_words)))))


def get_work_data():
    """
    initialize all the work data to the run
    """
    corpus = dependency_treebank.parsed_sents()
    train, test = train_test_split(corpus, test_size=0.1)

    words = _build_dict(corpus, WORD)
    tags = _build_dict(corpus, POS)

    return train, test, words, tags


def _get_category_index(mapping, u, v):
    """
    helper function for extracting the dict index of a word/tag
    """
    return mapping[u] * len(mapping) + mapping[v]


def feature_function(u, v, tree, words, tags):
    """
    the feature function for a tree object
    """
    return [_get_category_index(words, u[WORD], v[WORD]), _get_category_index(tags, u[POS], v[POS])]


def _tree2vec(tree, words, tags):
    """
    create a weights vector (Counter object) from a dependency graph
    """
    vec = defaultdict(int)
    nodes = tree.nodes
    for head in tree.nodes.values():
        tails = list(head[TAILS].values())
        tails = tails[0] if tails else tails
        for tail_idx in tails:
            word_idx, tag_idx = feature_function(head, nodes[tail_idx], tree, words, tags)
            vec[word_idx] += 1
            vec[tag_idx] += 1
    return Counter(vec)


def _arcs2vec(arcs, tree, words, tags):
    """
    create a weights vector (Counter object) from arcs returned by the max tree algorithm
    """
    vec = defaultdict(int)
    nodes = tree.nodes
    for node in arcs.values():
        head, tail = nodes[node.head], nodes[node.tail]
        word_idx, tag_idx = feature_function(head, tail, tree, words, tags)
        vec[word_idx] += 1
        vec[tag_idx] += 1
    return Counter(vec)


def _arc_init(head, tail, weights, tree, words, tags):
    """
    helper function - creates a single arc
    """
    return ARC(head[0], tail[0], weights[feature_function(head[1], tail[1], tree, words, tags)].sum())


def _get_all_arcs(w, tree):
    """
    helper function - creates list of all possible arcs.
    """
    nodes = tree.nodes.items()
    return [_arc_init(head, tail, w, tree, words, tags) for \
            head, tail in filter(lambda p: p[0] != p[1], product(nodes, nodes))]


def perceptron_train(data, words, tags, n_iters=2):
    """
    function to train the perceptron, returns the fitted weights.
    """
    # initialize:
    n = len(data)
    d = np.power(len(words), 2) + np.power(len(tags), 2)
    w = np.zeros(d)
    cumulative_change = Counter({})

    # train:
    for r in range(n_iters):
        with tqdm.tqdm(np.random.permutation(data)) as tepoch:
            for golden_tree in tepoch:
                # get the golden tree vector:
                golden_vec = _tree2vec(golden_tree, words, tags)

                # get the best tree vector:
                tepoch.set_postfix(status='finding max tree Chu Liu...')
                best_tree_vec = _arcs2vec(max_spanning_arborescence_nx(_get_all_arcs(w, golden_tree)),
                                          golden_tree, words, tags)

                tepoch.set_postfix(status='updating weights')
                # update changes, and weights:
                change = (golden_vec - best_tree_vec)
                cumulative_change += change

                for idx, val in change.items():
                    w[idx] += val

    # extract final weights:
    w = np.zeros_like(w)
    for idx, val in cumulative_change.items():
        w[idx] += val / (n * n_iters)

    return w


def perceptron_predict(weights, tree):
    return [(arc.head, arc.tail) for arc in max_spanning_arborescence_nx(_get_all_arcs(weights, tree)).values()]


def accuracy(perceptron, data):
    acc = 0
    with tqdm.tqdm(data) as tepoch:
        for tree in tepoch:
            tree_acc = 0
            prediction_tree = perceptron_predict(perceptron, tree)
            n = len(prediction_tree)
            for i, node in tree.nodes.items():
                tails = list(node[TAILS].values())
                if tails:
                    found_arcs = [(i, j) for j in tails[0] if (i, j) in prediction_tree]
                    [prediction_tree.remove(p) for p in found_arcs]
                    tree_acc += len(found_arcs)
            tree_acc /= n
            tepoch.set_postfix(tree_accuracy=tree_acc)
            acc += tree_acc
    return np.round((acc / len(data)) * 100, 3)


if __name__ == '__main__':
    # get work data:
    train, test, words, tags = get_work_data()

    # train model:
    perceptron = perceptron_train(train, words, tags)

    # evaluating model:
    print(f'accuracy on test is {accuracy(perceptron, test)}%')
