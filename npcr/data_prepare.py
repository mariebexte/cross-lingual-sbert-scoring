import sys
import npcr.utils as utils
import npcr.reader as reader
import random
import logging

import numpy as np
import pandas as pd

from copy import deepcopy


logger = utils.get_logger("Prepare data ...")

def prepare_sentence_data_adversarial(df_train, df_val, df_test, max_num, base_model, col_answer, col_score, col_embedding='input_ids', col_attention_mask='attention_mask'):
    
    df_train, df_val, df_test = reader.get_data(df_train=df_train, df_val=df_val, df_test=df_test, max_length=max_num, base_model=base_model, col_score=col_score, col_answer=col_answer)

    logger.info('Statistics:')
    logger.info('  train X shape: ' + str(np.array(df_train[col_embedding]).shape))

    if df_val is not None:

        logger.info('  dev X shape:   ' + str(np.array(df_val[col_embedding]).shape))

    if df_test is not None:

        logger.info('  test X shape:  ' + str(np.array(df_test[col_embedding]).shape))

    logger.info('  train Y shape: ' + str(np.array(df_train[col_score]).shape))

    if df_val is not None:

        logger.info('  dev Y shape:   ' + str(np.array(df_val[col_score]).shape))

    if df_test is not None:

        logger.info('  test Y shape:  ' + str(np.array(df_test[col_score]).shape))

    return df_train, df_val, df_test


def get_training_pairs(df_train, col_prompt, col_score, num_training_pairs=None, training_within_prompt=True, training_with_same_score=False, col_embedding='input_ids', col_mask='attention_mask', min_label=None, max_label=None):

    # To keep track of instances for which no pairs can be built
    train_without_partner = 0

    features_train = []
    masks_train = []
    y_train = []

    # Scale scores to interval of 0 to 1
    col_score_scaled = 'holistic_score_scaled'
    df_train[col_score_scaled] = reader.get_model_friendly_scores_adversarial(df_train[col_score], min_label=min_label, max_label=max_label, prompts_array=df_train[col_prompt])

    # Pair only within the same prompt
    if training_within_prompt:

        logging.info('Building training pairs within prompt!')

        for prompt, df_prompt in df_train.groupby(col_prompt):

            # Pair element i with i+1, i+2 and i+3
            for i in range(len(df_prompt) - 3):

                for j in range(1, 4):

                    if training_with_same_score:

                        features_train.append((df_train[col_embedding].iloc[i], df_train[col_embedding].iloc[i+j]))
                        masks_train.append((df_train[col_mask].iloc[i], df_train[col_mask].iloc[i+j]))
                        y_train.append(df_train[col_score_scaled].iloc[i] - df_train[col_score_scaled].iloc[i+j])
                    
                    else:

                        if not (df_train[col_score_scaled].iloc[i] == df_train[col_score_scaled].iloc[i+j]):

                            features_train.append((df_train[col_embedding].iloc[i], df_train[col_embedding].iloc[i+j]))
                            masks_train.append((df_train[col_mask].iloc[i], df_train[col_mask].iloc[i+j]))
                            y_train.append(df_train[col_score_scaled].iloc[i] - df_train[col_score_scaled].iloc[i+j])
            
            # This method leaves the last three elements of the dataframe without pair
            train_without_partner += 3
    
    # Pair only across prompts (but up to same number of training instances)
    else:

        logging.info('Building training pairs cross-prompt!')

        # Build resource to pick reference examples from
        prompt_dict = {}

        for prompt, df_prompt in df_train.groupby(col_prompt):

            # Need at least three examples for one pairing step
            if len(df_prompt) > 2:

                prompt_dict[prompt] = deepcopy(df_prompt).reset_index()

        for idx, row in df_train.iterrows():

            current_prompt = row[col_prompt]
            all_prompts = deepcopy(list(prompt_dict.keys()))

            # Might already be deleted from previous pairings
            if current_prompt in all_prompts:

                all_prompts.remove(current_prompt)

            if len(all_prompts) > 0:

                random.shuffle(all_prompts)
                # Pick which prompt reference answers should come from
                pairing_prompt = all_prompts[0]

                # Transpose in order to be able to pop an item
                df_pair = prompt_dict[pairing_prompt].T

                for j in range(3):

                    # Remove first object
                    if j==0:

                        row_pair = dict(df_pair.pop(0))

                    # But keep 2nd and 3rd (for analogy with how pairs are built within-prompt)
                    else:

                        # Have to subtract one, because first element was popped
                        row_pair = dict(df_pair.iloc[:, j-1])

                    if training_with_same_score:

                        features_train.append((row[col_embedding], row_pair[col_embedding]))
                        masks_train.append((row[col_mask], row_pair[col_mask]))
                        y_train.append(row[col_score_scaled] - row_pair[col_score_scaled])

                    else:

                        if not (row[col_score_scaled] == row_pair[col_score_scaled]):

                            features_train.append((row[col_embedding], row_pair[col_embedding]))
                            masks_train.append((row[col_mask], row_pair[col_mask]))
                            y_train.append(row[col_score_scaled] - row_pair[col_score_scaled])                            
                
                ## Check if there are enough items still left for another round of reference example pairing
                # Re-transpose
                df_pair = df_pair.T

                if len(df_pair) > 2:

                    # Otherwise index will build up over time
                    df_pair.pop('index')
                    prompt_dict[pairing_prompt] = df_pair.reset_index()

                else:

                    print('removing prompt', pairing_prompt)
                    prompt_dict.pop(pairing_prompt)
            
            else:

                # print('No more prompts to pair for this index! ', idx)
                train_without_partner += 1


    logging.info(str(train_without_partner) + " training examples are without partner!")

    features_train = np.array(features_train)
    masks_train = np.array(masks_train)
    y_train = np.array(y_train)

    # Push training pair labels from range of -1 to 1 into range 0 to 1
    y_train = reader.offcenter_labels(y_train)

    # A limit might have been placed on the number of training pairs
    if num_training_pairs is not None:

        if len(features_train) < num_training_pairs:

            logging.info("Stopping, because " + str(num_training_pairs) + ' were requested, but there are only ' + str(len(features_train)) + '!')
            sys.exit(0)
        
        features_train = features_train[0:num_training_pairs]
        masks_train = masks_train[0:num_training_pairs]
        y_train = y_train[0:num_training_pairs]

    return features_train, masks_train, y_train


def get_inference_pairs(df, df_ref, col_prompt, col_score, col_id, example_size=None, col_embedding='input_ids', col_mask='attention_mask', random_state=3456478, min_label=None, max_label=None):

    # Build pairs
    features = []
    masks = []
    y_goal = []
    y_example = []

    # Scale scores of reference data to range from 0 to 1
    col_scaled_score = 'holistic_score_scaled'
    df_ref[col_scaled_score] = reader.get_model_friendly_scores_adversarial(df_ref[col_score], min_label=min_label, max_label=max_label, prompts_array=df_ref[col_prompt])

    # Build cross-product of df and reference instances
    df_cartesian_full = df.merge(df_ref, how='cross', suffixes=('', '_ref'))
    cartesians = []

    if example_size is not None:

        for item in df[col_id]:
            df_cartesian_item = df_cartesian_full[df_cartesian_full[col_id] == item]
            cartesians.append(df_cartesian_item.sample(example_size))

    df_cartesian = pd.concat(cartesians).reset_index()
    
    features = features + list(df_cartesian.apply(lambda row: (row[col_embedding], row[col_embedding+'_ref']), axis='columns'))
    masks = masks + list(df_cartesian.apply(lambda row: (row[col_mask], row[col_mask+'_ref']), axis='columns'))
    y_goal = y_goal + list(df[col_score])

    # Rescale reference labels to label range of test/val instance
    y_example_prompt = utils.rescale_tointscore_adversarial(df_cartesian[col_scaled_score], min_label=min_label, max_label=max_label, prompts_array=df_cartesian[col_prompt])
    y_example = y_example + list(y_example_prompt)

    # Restructure for compatibility
    y_goal = [[value] for value in y_goal]
    y_example = [[value] for value in y_example]

    features = np.array(features)
    masks = np.array(masks)
    y_example = np.array(y_example)
    y_goal = np.array(y_goal)

    return features, masks, y_example, y_goal
