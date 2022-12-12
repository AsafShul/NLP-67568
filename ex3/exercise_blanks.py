import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import platform
from time import sleep

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

BASE = 'base'
NEG_POLAR = 'negated_polarity'
RARE_WORDS = 'rare_words'

W2V_PATH = "w2v_dict.pkl"
W2V_SEQ_PATH = "w2v_dict_seq.pkl"

TRAIN = "train"
VAL = "val"
TEST = "test"

NO_ACCELERATION = 'cpu'
MACOS_GPU_ACCELERATION = 'mps'
NVIDIA_GPU_ACCELERATION = 'cuda'

RES_DF_COLS = ['Train_loss', 'Train_acc', 'Val_loss', 'Val_acc']
PLOT_TITLE = '{}: {} plot for {} epochs'


# ------------------------------------------ Helper methods and classes --------------------------

def _running_on_mac():
    """
    :return: true if running on macOs False otherwise.
    """
    return platform.system() == 'Darwin'


def get_available_device(log=False):
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    available_device = NO_ACCELERATION
    if USE_ACCELERATION:
        if _running_on_mac() and torch.backends.mps.is_available():
            available_device = MACOS_GPU_ACCELERATION
        elif torch.cuda.is_available():
            available_device = NVIDIA_GPU_ACCELERATION
        if log:
            print(f'Using device: "{available_device}" '
                  f'{"(gpu acceleration)." if available_device != NO_ACCELERATION else "."}')
    return torch.device(available_device)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False, w2v_path=W2V_PATH):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param w2v_path: res pickle file name
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    return np.mean([word_to_vec[w] if w in word_to_vec else np.zeros(embedding_dim) for w in sent.text], axis=0)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = np.zeros(size)
    vec[ind] = 1
    return vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    mean = np.zeros(len(word_to_ind))
    for word in sent.text:
        mean[word_to_ind[word]] += 1
    return mean / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    idx = 0
    idx_dict = {}
    for word in words_list:
        if word not in idx_dict.keys():
            idx_dict[word] = idx
            idx += 1
    return idx_dict


# todo check default
def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """

    vec = [word_to_vec[w] if w in word_to_vec else np.zeros(embedding_dim) for w in sent.text]
    if len(vec) < seq_len:
        return np.vstack([np.array(vec), np.zeros((seq_len - len(vec), embedding_dim))])
    return vec[:seq_len]


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


def filter_sentences(sentences, func, *args):
    return list(np.array(sentences)[func(sentences, *args)])


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=50, embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """
        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)

        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preparation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}

        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding
            cache = os.path.exists(W2V_SEQ_PATH)

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list, not cache,
                                                                            w2v_path=W2V_SEQ_PATH),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            cache = os.path.exists(W2V_PATH)

            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list, not cache),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.name = f"LSTM"
        self.device = get_available_device(True)
        self.sigmoid = nn.Sigmoid()
        self.alpha = 0.5
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True,
                            num_layers=n_layers, dropout=dropout, device=self.device, batch_first=True)
        self.linear = nn.Linear(2 * hidden_dim, 1, device=self.device)

    def forward(self, text):
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(text)  # lstm with input, hidden, and internal state
        return self.linear(torch.hstack([hn[0], hn[1]]))

    def predict(self, text):
        return self.sigmoid(self.forward(text)) > self.alpha


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.name = f"Log-Linear ({'one-hot' if embedding_dim > 300 else 'W2V'})"
        self.device = get_available_device(True)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(embedding_dim, 1, device=self.device)
        self.alpha = 0.5

    def forward(self, x):
        return self.linear(x.to(torch.float32).to(self.device))

    def predict(self, x):
        return self.sigmoid(self.forward(x)) > self.alpha


# ------------------------- training functions -------------
def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return (preds == y).sum() / preds.shape[0]


def train_epoch(model, data_iterator, optimizer, criterion, n):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param n: the epoch iteration number
    :param model: the model we're currently training
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    epoch_loss, epoch_acc = [], []
    batch_num = len(data_iterator)
    with tqdm.tqdm(data_iterator, unit="batch") as tepoch:
        for i, (data, target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch [{n}]")
            data, target = data.to(torch.float32).to(model.device), target.to(torch.float32).to(model.device)

            # forward:
            optimizer.zero_grad()
            output = model(data).reshape((-1,))

            # backward:
            iter_loss = criterion(output, target)
            iter_loss.backward()
            optimizer.step()

            # metrics:
            accuracy = binary_accuracy(get_predictions_for_data(model, output), target)  # todo api
            epoch_loss.append(iter_loss.item())
            epoch_acc.append(accuracy.item())

            if i < batch_num - 1:
                tepoch.set_postfix(loss=iter_loss.item(), accuracy=100. * accuracy.item())
            else:
                tepoch.set_postfix(loss=np.mean(epoch_loss), accuracy=100. * np.mean(epoch_acc))
    return np.mean(epoch_loss), np.mean(epoch_acc)


def evaluate(model, data_iterator, criterion, n, seq='Val'):
    """
    evaluate the model performance on the given data
    :param seq: train / test / val string
    :param n: epoch iteration number
    :param model: one of our models.
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    loss, acc = [], []
    batch_num = len(data_iterator)
    with torch.no_grad():
        with tqdm.tqdm(data_iterator, unit="batch") as tepoch:
            for i, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"{seq}-eval-Epoch [{n}] ")
                data, target = data.to(torch.float32).to(model.device), target.to(torch.float32).to(model.device)

                # forward:
                output = model(data).reshape((-1,))
                iter_loss = criterion(output, target)
                accuracy = binary_accuracy(get_predictions_for_data(model, output), target)
                loss.append(iter_loss.item())
                acc.append(accuracy.item())

                if i < batch_num - 1:
                    tepoch.set_postfix(loss=iter_loss.item(), accuracy=100. * accuracy.item())
                else:
                    tepoch.set_postfix(loss=np.mean(loss), accuracy=100. * np.mean(acc))

        return np.mean(loss), np.mean(acc)


def get_predictions_for_data(model, output):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param output: model output for current outputs
    :return:
    """
    return (model.sigmoid(output) > model.alpha).to(torch.float32).reshape((-1,))


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones([model.v], device=model.device))
    criterion = nn.BCEWithLogitsLoss()  # todo pos weights?
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_data_iterator = data_manager.get_torch_iterator(TRAIN)
    val_data_iterator = data_manager.get_torch_iterator(VAL)
    test_data_iterator = data_manager.get_torch_iterator(TEST)

    df = pd.DataFrame(np.nan, index=range(n_epochs), columns=RES_DF_COLS)

    for epoch in range(1, n_epochs + 1):
        print('\n', '=' * 40, f'epoch [{epoch}]', '=' * 40)
        sleep(0.1)
        epoch_train_loss, epoch_train_acc = train_epoch(model, train_data_iterator, optimizer, criterion, epoch)
        epoch_val_loss, epoch_val_acc = evaluate(model, val_data_iterator, criterion, epoch)
        df.loc[epoch] = [epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc]

    plot_res(df, model, 'acc', n_epochs)
    plot_res(df, model, 'loss', n_epochs)
    _, test_acc = evaluate(model, test_data_iterator, criterion, n_epochs + 1)
    neg_acc = test_spacial_case(model, criterion, data_manager, NEG_POLAR)
    rare_acc = test_spacial_case(model, criterion, data_manager, RARE_WORDS)
    print('=' * 50)
    print(f'{model.name} Test results: ')
    print(f'\t accuracy (test) = {test_acc}')
    print(f'\t accuracy (neg ) = {neg_acc}')
    print(f'\t accuracy (rare) = {rare_acc}')
    print('=' * 50)


def test_spacial_case(model, criterion, data_manager, mode):
    if mode == NEG_POLAR:
        idx = data_loader.get_negated_polarity_examples(data_manager.sentences[TEST])
    elif mode == RARE_WORDS:
        idx = data_loader.get_rare_words_examples(data_manager.sentences[TEST], data_manager.sentiment_dataset)
    else:
        return
    spacial_sentences = np.array(data_manager.sentences[TEST])[idx]
    loader = DataLoader(OnlineDataset(spacial_sentences, data_manager.sent_func, data_manager.sent_func_kwargs))
    _, acc = evaluate(model, loader, criterion, '*spacial*')
    return acc


def plot_res(df, model, metric, n_epochs, save=True):
    ax = df[[c for c in RES_DF_COLS if metric in c]].plot(title=PLOT_TITLE.format(model.name, metric, n_epochs))
    ax.set_xlabel("epoch num")
    ax.set_ylabel(metric)
    if save: ax.get_figure().savefig(f'{model.name.replace(" ", "")}_{metric}.jpg')
    plt.show()


def train_log_linear_with_one_hot(lr=0.01, n_epochs=20, batch_size=64, weight_decay=0.001):  # todo api
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(batch_size=batch_size)
    model = LogLinear(data_manager.get_input_shape()[0])
    train_model(model, data_manager, n_epochs, lr, weight_decay=weight_decay)


def train_log_linear_with_w2v(lr=0.01, n_epochs=20, batch_size=64, weight_decay=0.001, embedding_dim=300):  # todo api
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, embedding_dim=embedding_dim, batch_size=batch_size)
    model = LogLinear(data_manager.get_input_shape()[0])
    train_model(model, data_manager, n_epochs, lr, weight_decay=weight_decay)


def train_lstm_with_w2v(lr=0.001, n_epochs=4, batch_size=64,
                        weight_decay=0.0001, embedding_dim=300, n_layers=1, dropout=0.5):
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE, embedding_dim=embedding_dim, batch_size=batch_size)
    model = LSTM(embedding_dim=embedding_dim, hidden_dim=100, n_layers=n_layers, dropout=dropout)
    train_model(model, data_manager, n_epochs, lr, weight_decay=weight_decay)


if __name__ == '__main__':
    USE_ACCELERATION = False

    train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()
