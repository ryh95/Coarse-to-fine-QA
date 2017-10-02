import json
import logging

import numpy as np

#  change as you like
file_to_split = '/home/ryh/dataset/wiki-reading/validation-00000-of-00015.json'
file_sample = 10
file_num = 3
early_stop_num = 100000

full_match_num = 0
partial_match_num = 0
none_match_num = 0

correct_sample_num = 0

full_match_dict = {}
partial_match_dict = {}
none_match_dict = {}


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

            if len(d['full_match_answer_location']) > 0:
                full_match_dict[full_match_num] = d
                full_match_num += 1


            if len(d['full_match_answer_location']) == 0 and len(d['answer_location']) > 0:
                partial_match_dict[partial_match_num] = d
                partial_match_num += 1

            if len(d['full_match_answer_location']) == 0 and len(d['answer_location']) == 0:
                none_match_dict[none_match_num] = d
                none_match_num += 1

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

    # print partial_match_num
    # print float(partial_match_num) / N

    for i in range(file_num):
        f_full = open('full-'+str(i)+'.json','w')
        f_partial = open('partial-'+str(i)+'.json','w')
        f_none = open('none-'+str(i)+'.json','w')

        full_sample_indices = np.random.choice(full_match_num,file_sample,replace=False)
        partial_sample_indices = np.random.choice(partial_match_num,file_sample,replace=False)
        none_match_sample_indices = np.random.choice(none_match_num,file_sample,replace=False)


        full_match_output = {}
        partial_match_output = {}
        none_match_output = {}

        for idx,indice in enumerate(full_sample_indices):
            full_match_output[idx] = full_match_dict[indice]

        for idx,indice in enumerate(partial_sample_indices):
            partial_match_output[idx] = partial_match_dict[indice]

        for idx,indice in enumerate(none_match_sample_indices):
            none_match_output[idx] = none_match_dict[indice]

        json.dump(full_match_output,f_full)
        logger.info("{}/{} dumps into file {}".format(file_sample,full_match_num,i))

        json.dump(partial_match_output,f_partial)
        logger.info("{}/{} dumps into file {}".format(file_sample,partial_match_num,i))

        json.dump(none_match_output,f_none)
        logger.info("{}/{} dumps into file {}".format(file_sample, none_match_num, i))

        f_full.close()
        f_partial.close()
        f_none.close()