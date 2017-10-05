import json
import logging
from os.path import join

import numpy as np

# change as you like
# since dataset is huge, use validation to train and test
file_to_split = join("../","data","original_data","validation-00000-of-00015.json")
train_path = join("../","data","train","train_set.json")
test_path = join("../","data","test","test_set.json")
train_sample = 10000
test_sample = 400
early_stop_num = 100000

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

    with open(train_path, 'w') as f_train,\
        open(test_path,'w') as f_test:
        sample_indices = np.random.choice(N, train_sample+test_sample, replace=False)

        for indice in sample_indices[:train_sample]:
            f_train.write(correct_sample[indice])
        for indice in sample_indices[train_sample:]:
            f_test.write(correct_sample[indice])