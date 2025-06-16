import os
import sys
import torch
import logging
import npcr.data_prepare as data_prepare

import numpy as np
import pandas as pd
import torch.utils.data as Data

from npcr.metrics import *
from tqdm import tqdm
from copy import deepcopy
from npcr.utils import rescale_tointscore_adversarial, get_logger, write_classification_stats, SCORES

from config import NPCR_NUM_TEST_PAIRS


logger = get_logger("Evaluate stats")


class Evaluator_opti_adversarial():


    def __init__(self, out_dir, model_name, features_dev, masks_dev, dev_y_example, dev_y_goal, example_size, min_label=None, max_label=None, prompts_array=None, save_model=True, save_stats=True, suffix='', patience=None):

        self.out_dir = out_dir
        self.modelname = model_name

        self.features_dev = features_dev
        self.masks_dev = masks_dev
        self.dev_y_example = dev_y_example
        self.dev_y_goal = dev_y_goal

        self.example_size = example_size

        self.best_dev = [10000, -1, -1, -1, -1]
        self.dev_loader = self.init_data_loader()

        # To track and write training stats to file
        self.dict_performance = {}
        self.dict_performance_idx = 0

        self.save_model = save_model
        self.save_stats = save_stats
        self.suffix = suffix

        self.min_label = min_label
        self.max_label = max_label

        self.patience = patience
        self.stop = False

        if patience is None:

            self.patience = None

        if min_label is None:

            self.min_label = None

        if max_label is None:

            self.max_label = None

        # List of prompt info on the validation/test instances
        self.prompts_array = prompts_array

        if prompts_array is None:

            self.prompts_array = None


    def calc_correl(self, dev_pred):

        self.dev_pr = pearson(self.dev_y_goal.squeeze(), dev_pred.squeeze())
        self.dev_spr = spearman(self.dev_y_goal, dev_pred)

    def calc_kappa(self, dev_pred, weight='quadratic'):

        self.dev_qwk = kappa(self.dev_y_goal, dev_pred, weight)

    def calc_rmse(self, dev_pred):

        self.dev_rmse = root_mean_square_error(self.dev_y_goal, dev_pred)


    def print_info(self, epoch):

        logger.info('[DEV]   LOSS:  %.3f, QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f, (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f, %.3f)' % (
            self.dev_loss, self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse, self.best_dev_epoch,
            self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[4]))

        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')

        self.dict_performance[self.dict_performance_idx] = {'epoch': epoch, 'val loss': self.dev_loss, 'val QWK': self.dev_qwk, 'val PRS': self.dev_pr,\
            'val SPR': self.dev_spr, 'val RMSE': self.dev_rmse, 'val best epoch': self.best_dev_epoch}
        
        self.dict_performance_idx += 1

        self.df_performance = pd.DataFrame.from_dict(self.dict_performance, orient='index')
        
        if self.save_stats:

            self.df_performance.to_csv(os.path.join(self.out_dir, 'training_stats' + self.suffix + '.csv'))


    def evaluate(self, model, epoch, until, print_info=True, return_preds=False):

        dev_pred_int = []
        dev_pred_raw = []

        losses = []

        for step, (batch_dev_x0, batch_dev_x1, batch_dev_mask_x0, batch_dev_mask_x1, batch_example_s) in tqdm(enumerate(self.dev_loader), total=len(self.dev_loader)):
            
            if self.prompts_array is not None:

                batch_prompt = self.prompts_array[step]
                batch_prompt_array = [batch_prompt] * len(batch_example_s)
            
            else:

                batch_prompt = None
                batch_prompt_array = None

            # If a limitation was placed on the number of reference examples, this will take effect here
            if until > len(batch_dev_x0):

                logging.info('Evaluation with more reference examples than previously built was requested! Built: ' + str(len(batch_dev_x0)) + ' Requested: ' + str(until))
            
            batch_dev_x0 = batch_dev_x0[0:until]
            batch_dev_x1 = batch_dev_x1[0:until]
            batch_dev_mask_x0 = batch_dev_mask_x0[0:until]
            batch_dev_mask_x1 = batch_dev_mask_x1[0:until]
            batch_example_s = batch_example_s[0:until]
            
            dev_y_pred = model(batch_dev_x0.cuda(), batch_dev_x1.cuda(), batch_dev_mask_x0.cuda(), batch_dev_mask_x1.cuda())
            
            
            # Rescale predicted deviations to range of possible DEVIATIONS within label range of current prompt
            dev_y_pred_unscaled = rescale_tointscore_adversarial(dev_y_pred.detach().cpu().numpy(), min_label=self.min_label, max_label=self.max_label, prompts_array=batch_prompt_array, differences=True)
            
            true_score = self.dev_y_goal[step]
            true_deviation = (batch_example_s.squeeze().cpu() * -1 ) + true_score
            losses.append(mean_square_error(dev_y_pred_unscaled.squeeze(), true_deviation))
            
            # Add predicted deviation to scores of reference example
            dev_pred_group = dev_y_pred_unscaled + batch_example_s.detach().cpu().numpy()
            
            # One batch equals one instance + all its reference examples, therefore average to get score for this instance
            dev_pred = np.mean(dev_pred_group)
            dev_pred_i = np.around(dev_pred).astype(int)
            dev_pred_i = np.array([dev_pred_i])

            # Reel in out of bounds predictions
            if (self.min_label is not None) and (self.max_label is not None):

                dev_pred_i[dev_pred_i > self.max_label] = self.max_label
                dev_pred_i[dev_pred_i < self.min_label] = self.min_label

            else:
                dev_pred_i[dev_pred_i > SCORES[batch_prompt][1]] = SCORES[batch_prompt][1]
                dev_pred_i[dev_pred_i < SCORES[batch_prompt][0]] = SCORES[batch_prompt][0]

            dev_pred_int.append(dev_pred_i)
            dev_pred_raw.append(dev_pred)

        dev_pred_int = np.array(dev_pred_int)
        dev_pred_raw = np.array(dev_pred_raw)

        avg_loss = sum(losses)/len(losses)
        self.dev_loss = avg_loss

        self.calc_correl(dev_pred_int)
        self.calc_kappa(dev_pred_int)
        self.calc_rmse(dev_pred_int)

        if self.dev_loss < self.best_dev[0]:

            self.best_dev = [self.dev_loss, self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse]
            self.best_dev_epoch = epoch

            if self.save_model:

                torch.save(model, self.out_dir + '/' + self.modelname)
        
        # No improvement, check if patience exceeded
        else:

            if self.patience is not None and ((epoch - self.best_dev_epoch) >= self.patience):

                self.stop = True

        if print_info:

            self.print_info(epoch)

        if return_preds:

            return dev_pred_int, dev_pred_raw


    def init_data_loader(self):

        dev_x0 = [j[0] for j in self.features_dev]
        dev_x1 = [j[1] for j in self.features_dev]
        dev_x0 = torch.LongTensor(np.array(dev_x0))
        dev_x1 = torch.LongTensor(np.array(dev_x1))

        dev_masks_x0 = [m[0] for m in self.masks_dev]
        dev_masks_x1 = [m[1] for m in self.masks_dev]
        dev_masks_x0 = torch.Tensor(np.array(dev_masks_x0))
        dev_masks_x1 = torch.Tensor(np.array(dev_masks_x1))

        dev_example_s = torch.Tensor(self.dev_y_example)

        dev_dataset = Data.TensorDataset(dev_x0, dev_x1, dev_masks_x0, dev_masks_x1, dev_example_s)

        dev_loader = Data.DataLoader(

            dataset=dev_dataset,
            batch_sampler=Data.BatchSampler(
                Data.SequentialSampler(data_source=dev_dataset), batch_size=self.example_size, drop_last=False
            ),
            num_workers=2
        )

        return dev_loader


def evaluate_finetuned_model(df_test, df_ref, col_id, col_prompt, col_answer, col_score, model_path, base_model, target_path, max_num, example_size=NPCR_NUM_TEST_PAIRS, min_label=None, max_label=None, suffix=''):

    logging.info('Evaluation: min score is:\t' + str(min_label))
    logging.info('Evaluation: max score is:\t' + str(max_label))

    if example_size is None:
        example_size = len(df_ref)

    if not os.path.exists(target_path):

        os.mkdir(target_path)

    df_ref, df_val, df_test = data_prepare.prepare_sentence_data_adversarial(df_train=df_ref, df_val=None, df_test=df_test, col_answer=col_answer, col_score=col_score, max_num=max_num, base_model=base_model)
    features_test, masks_test, y_test_example, y_test_goal = data_prepare.get_inference_pairs(df=df_test, df_ref=df_ref, col_id=col_id, col_prompt=col_prompt, col_score=col_score, example_size=example_size, min_label=min_label, max_label=max_label)

    model = torch.load(model_path, weights_only=False)
    model.cuda()

    evl = Evaluator_opti_adversarial(out_dir=target_path, model_name=None, features_dev=features_test, masks_dev=masks_test,\
        dev_y_example=y_test_example, dev_y_goal=y_test_goal, min_label=min_label, max_label=max_label, prompts_array=df_test[col_prompt], example_size=example_size, save_model=False, save_stats=False, suffix=suffix)

    model.eval()

    with torch.no_grad():

        y_pred, y_pred_raw = evl.evaluate(model, 1, example_size, True, True)

    y_pred = y_pred.squeeze()
    y_pred_raw = y_pred_raw.squeeze()
    y_true = evl.dev_y_goal.squeeze()
    write_classification_stats(output_dir=target_path, y_true=y_true, y_pred=y_pred, y_true_diff=None, y_pred_diff=None, suffix=suffix)

    df_test_copy = deepcopy(df_test)
    df_test_copy['pred'] = y_pred
    df_test_copy['pred_raw'] = y_pred_raw
    df_test_copy.to_csv(os.path.join(target_path, 'preds' + suffix + '.csv'))

    return y_true, y_pred
    