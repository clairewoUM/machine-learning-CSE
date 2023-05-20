"""HW2: Naive Bayes for Classifying SPAM."""

from typing import Tuple

import numpy as np
import math

def hello():
    print('Hello from naive_bayes_spam.py')

def train_naive_bayes(X: np.ndarray, Y: np.ndarray,
                      ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes probabilities for logit x being each class.

    Inputs:
      - X: Numpy array of shape (num_mails, vocab_size) that represents emails.
        The (i, j)th entry of X represents the number of occurrences of the
        j-th token in the i-th document.
      - Y: Numpy array of shape (num_mails). It includes 0 (non-spam) or 1 (spam).
    Returns: A tuple of
      - mu_spam: Numpy array of shape (vocab_size). mu value for SPAM mails.
      - mu_non_spam: Numpy array of shape (vocab_size). mu value for Non-SPAM mails.
      - phi: the ratio of SPAM mail from the dataset email.
    """
    num_mails, vocab_size = X.shape
    mu_spam = None
    mu_non_spam = None
    phi = 0.0

    spam_counts = list(Y).count(1)
    non_spam_counts = list(Y).count(0)
    phi = spam_counts/num_mails
    spam_wordj_counts = X[np.where(Y == 1),:][0]
    non_spam_wordj_counts = X[np.where(Y == 0),:][0]
    mu_spam = (np.sum(spam_wordj_counts, axis = 0) + 1.0) / (np.sum(spam_wordj_counts)+ 1.0 * vocab_size)
    mu_non_spam = (np.sum(non_spam_wordj_counts, axis = 0) + 1.0) / (np.sum(non_spam_wordj_counts)+ 1.0 * vocab_size)    

    return mu_spam, mu_non_spam, phi

def test_naive_bayes(X: np.ndarray,
                     mu_spam: np.ndarray,
                     mu_non_spam: np.ndarray,
                     phi: float,
                     ) -> np.ndarray:
    """Classify whether the emails in the test set is SPAM.

    Inputs:
      - X: Numpy array of shape (num_mails, vocab_size) that represents emails.
        The (i, j)th entry of X represents the number of occurrences of the
        j-th token in the i-th document.
      - mu_spam: Numpy array of shape (vocab_size). mu value for SPAM mails.
      - mu_non_spam: Numpy array of shape (vocab_size). mu value for Non-SPAM mails.
      - phi: the ratio of SPAM mail from the dataset email.
    Returns:
      - pred: Numpy array of shape (num_mails). Mark 1 for the SPAM mails.
    """
    pred = np.zeros(X.shape[0])

    log_spam_probs = np.log(phi)
    log_non_spam_probs = np.log(1.0-phi)
    log_word_given_spam_probs = X.dot(np.log(mu_spam))
    log_word_given_non_spam_probs = X.dot(np.log(mu_non_spam))

    log_ll_spam = log_spam_probs + log_word_given_spam_probs
    log_ll_non_spam = log_non_spam_probs + log_word_given_non_spam_probs

    pred = np.array([int(log_ll_spam[k] > log_ll_non_spam[k]) for k in range(X.shape[0])])    

    return pred


def evaluate(pred: np.ndarray, Y: np.ndarray) -> float:
    """Compute the accuracy of the predicted output w.r.t the given label.

    Inputs:
      - pred: Numpy array of shape (num_mails). It includes 0 (non-spam) or 1 (spam).
      - Y: Numpy array of shape (num_mails). It includes 0 (non-spam) or 1 (spam).
    Returns:
      - accuracy: accuracy value in the range [0, 1].
    """
    accuracy = np.mean((pred == Y).astype(np.float32))

    return accuracy


def get_indicative_tokens(mu_spam: np.ndarray,
                          mu_non_spam: np.ndarray,
                          top_k: int,
                          ) -> np.ndarray:
    """Filter out the most K indicative vocabs from mu.

    We will check the lob probability of mu's. Your goal is to return `top_k`
    number of vocab indices.

    Inputs:
      - mu_spam: Numpy array of shape (vocab_size). The mu value for
                 SPAM mails.
      - mu_non_spam: Numpy array of shape (vocab_size). The mu value for
                     Non-SPAM mails.
      - top_k: The number of indicative tokens to generate. A positive integer.
    Returns:
      - idx_list: Numpy array of shape (top_k), of type int (or int32).
                  Each index represent the vocab in vocabulary file.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    idx_list = np.zeros(top_k, dtype=np.int32)

    ind_tokens = np.log(mu_spam/mu_non_spam)
    idx_list = np.argsort(ind_tokens)[ :top_k]

    return idx_list
