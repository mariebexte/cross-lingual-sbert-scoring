# License: BSD 3 clause
"""
This module contains a bunch of evaluation metrics that can be used to
evaluate the performance of learners.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
"""

from __future__ import print_function, unicode_literals

import logging
import math
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from six import string_types
from six.moves import xrange as range
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from sklearn.metrics._scorer import _SCORERS as SCORERS


# Constants
_CORRELATION_METRICS = frozenset(['kendall_tau', 'spearman', 'pearson'])


def kappa(y_true, y_pred, weights=None):

    if weights == None:
        return cohen_kappa_score(y_true, y_pred) 
    
    else:
        return cohen_kappa_score(y_true, y_pred, weights=weights) 


def spearman(y_true, y_pred):
    """
    Calculate Spearman's rank correlation coefficient between ``y_true`` and
    ``y_pred``.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float

    :returns: Spearman's rank correlation coefficient if well-defined, else 0
    """
    
    ret_score = spearmanr(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0



def pearson(y_true, y_pred):
    """
    Calculate Pearson product-moment correlation coefficient between ``y_true``
    and ``y_pred``.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float

    :returns: Pearson product-moment correlation coefficient if well-defined,
              else 0
    """

    ret_score = pearsonr(y_true, y_pred)[0]
    return ret_score if not np.isnan(ret_score) else 0.0


def mean_square_error(y_true, y_pred):
    """
    Calculate the mean square error between predictions and true scores
    :param y_true: true score list
    :param y_pred: predicted score list
    return mean_square_error value
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = ((y_true-y_pred)**2).mean(axis=0)
    return float(mse)


def root_mean_square_error(y_true, y_pred):
    """
    Calculate the mean square error between predictions and true scores
    :param y_true: true score list
    :param y_pred: predicted score list
    return mean_square_error value
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = ((y_true-y_pred)**2).mean(axis=0)
    return float(math.sqrt(mse))
