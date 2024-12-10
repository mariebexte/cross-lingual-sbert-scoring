import os
import sys
import logging

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import classification_report, mean_squared_error, cohen_kappa_score


SCORES = {
    # EPIRLS
    'E011B14C': [0, 1], 'E011R02C': [0, 1], 'E011B09C': [0, 1], 'E011M11C': [0, 2], 'E011T02C': [0, 1], 
    'E011M13C': [0, 1], 'E011B03C': [0, 1], 'E011R11C': [0, 2], 'E011B08C': [0, 1], 'E011T10C': [0, 1], 
    'E011R05C': [0, 1], 'E011Z02C': [0, 2], 'E011M03C': [0, 1], 'E011T09C': [0, 1], 'E011R09C': [0, 1], 
    'E011M09C': [0, 1], 'E011M15C': [0, 1], 'E011B04C': [0, 1], 'E011B12C': [0, 1], 'E011R14C': [0, 1], 
    'E011T08C': [0, 1], 'E011M04C': [0, 1], 'E011M02C': [0, 1], 'E011T05C': [0, 2], 'E011M08C': [0, 1], 
    'E011Z14C': [0, 2], 'E011Z09C': [0, 1], 'E011Z12C': [0, 2], 'E011B13C': [0, 1], 'E011R08C': [0, 1], 
    'E011R16C': [0, 1], 'E011R15C': [0, 1], 'E011Z04C': [0, 1], 'E011T17C': [0, 1],
    # ASAP-SAS
    1: [0, 3], 2: [0, 3], 3: [0, 2], 4: [0, 2], 5: [0, 3], 6: [0, 3], 7: [0, 2], 8: [0, 2], 9: [0, 2], 10: [0, 2],
    '1': [0, 3], '2': [0, 3], '3': [0, 2], '4': [0, 2], '5': [0, 3], '6': [0, 3], '7': [0, 2], '8': [0, 2], '9': [0, 2], '10': [0, 2]
    }


def get_logger(name, level=logging.INFO, handler=sys.stdout,
        formatter='%(name)s - %(levelname)s - %(message)s'):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def rescale_tointscore_adversarial(scaled_scores, min_label=None, max_label=None, prompts_array=None, differences=False):

    if (min_label is None) and (max_label is None) and (prompts_array is None):
        logging.info('Cannot get model friendly scores: Neither min score, nor max score or prompts array was set!')
        sys.exit(0)

    if (min_label is not None) and (max_label is not None):
        # Scale to range of DEVIATION instead of pure scores
        if differences:
            min_label_copy = min_label
            max_label_copy = max_label
            min_label = min_label_copy - max_label_copy
            max_label = max_label_copy - min_label_copy
        # logging.info('Using min and max score to rescale labels! ' + str(min_label) + ' ' + str(max_label))
        int_scores = scaled_scores * (max_label - min_label) + min_label
        return int_scores
    
    else:
        # logging.info('Using prompt information to rescale labels! ' + str(prompts_array))    
        rescaled_scores = []
        for i in range(len(scaled_scores)):
            current_score = scaled_scores[i]
            current_prompt = prompts_array[i]
            min_label = SCORES[current_prompt][0]
            max_label = SCORES[current_prompt][1]
            # Scale to range of DEVIATION instead of pure scores
            if differences:
                min_label_copy = min_label
                max_label_copy = max_label
                min_label = min_label_copy - max_label_copy
                max_label = max_label_copy - min_label_copy
            rescaled_scores.append(current_score * (max_label - min_label) + min_label)
        
        return np.array(rescaled_scores)


# Write classification stats to file
def write_classification_stats(output_dir, y_true, y_pred, y_true_diff=None, y_pred_diff=None, suffix=''):

    with open(os.path.join(output_dir, 'stats' + suffix + '.csv'), 'w') as out_file:

        out_file.write(classification_report(y_true=y_true, y_pred=y_pred)+"\n\n")
        true_series = pd.Series(y_true, name='Actual')
        pred_series = pd.Series(y_pred, name='Predicted')
        out_file.write(str(pd.crosstab(true_series, pred_series))+"\n\n")
        out_file.write('QWK:\t' + str(cohen_kappa_score(y1 = y_true, y2 = y_pred, weights='quadratic')))
        pears_r, pears_p = pearsonr(y_true, y_pred)
        out_file.write('\nPearson:\t' + str(pears_r) + '\t(' + str(pears_p) + ')')

        # Partly for sanity (to compare against rmse in best validation epoch)
        if (y_true_diff is not None) and (y_pred_diff is not None):
            out_file.write('\n\nStats on raw predictions (on invididual reference examples):')
            out_file.write('\nRMSE:\t' + str(mean_squared_error(y_true = y_true_diff, y_pred = y_pred_diff, squared=False)))
            pearson_r, pearson_p = pearsonr(y_true_diff, y_pred_diff)
            out_file.write('\nPearson:\t' + str(pearson_r) + '\t(' + str(pearson_p) + ')')

            out_file.write('\nRMSE (smoothed):\t' + str(mean_squared_error(y_true = y_true_diff, y_pred = y_pred_diff, squared=False)))
            pearson_r, pearson_p = pearsonr(y_true_diff, y_pred_diff)
            out_file.write('\nPearson (smoothed):\t' + str(pearson_r) + '\t(' + str(pearson_p) + ')')
    