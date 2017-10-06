import json
import logging
from os.path import join
import os

import numpy as np

# change as you like
# since dataset is huge, use validation to train and test
import config

file_to_split = join("data","original_data","validation-00000-of-00015.json")
train_path = join("data","train")
test_path = join("data","test")
train_sample = config.TRAIN_NUM
test_sample = config.TEST_NUM
early_stop_num = config.EARLY_STOP_NUM

correct_sample_num = 0

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

with open(file_to_split,'r') as js_file:
    # raise json decode error
    # because it's not a standard json file
    # dataset = json.load(js_file)

    # remove json decode error
    correct_sample = []
    for i, line in enumerate(js_file):
        try:
            d = json.loads(line)

            correct_sample.append(line)

            correct_sample_num += 1
            # early stop to save memory
            if correct_sample_num >= early_stop_num:
                break
        except:
            print('Error on line', i + 1, ':\n', repr(line))

    # write into output file
    N = len(correct_sample)
    print N
    print correct_sample_num

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    with open(join(train_path,"train_set.json"), 'w') as f_train,\
        open(join(test_path,"test_set.json"),'w') as f_test:
        sample_indices = np.random.choice(N, train_sample+test_sample, replace=False)

        for indice in sample_indices[:train_sample]:
            f_train.write(correct_sample[indice])
        for indice in sample_indices[train_sample:]:
            f_test.write(correct_sample[indice])