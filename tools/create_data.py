import json
import logging

import numpy as np

#  change as you like
file_to_split = '/home/ryh/dataset/wiki-reading/validation-00000-of-00015.json'
file_sample = 500
file_num = 2
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

    for i in range(file_num):
        with open('validation-' + str(i) + '.json', 'w') as f:
            sample_indices = np.random.choice(N, file_sample, replace=False)

            for indice in sample_indices:
                f.write(correct_sample[indice])