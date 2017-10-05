import json
import random
from os.path import join
import cPickle

import numpy as np

from main import initialize_vocabulary, load_glove_embeddings

UNK = 2
placeholder_num = 100
placeholders = [u'PH_'+str(i) for i in range(placeholder_num)]
random.shuffle(placeholders)

vocab_list = initialize_vocabulary()
id2word = {idx:word for idx,word in  enumerate(vocab_list)}
word2id = {word:idx for idx,word in  enumerate(vocab_list)}

emb_size = 100
emb_path = join("data","dwr","glove.trimmed.{}.npz".format(emb_size))
embeddings = load_glove_embeddings(emb_path)

new_id = embeddings.shape[0]

def substitute(ids,tokens):
    global new_id
    global embeddings
    for i, (id, token) in enumerate(zip(ids, tokens)):
        if id == UNK:
            if token not in word2id:
                # distribute an id
                if len(placeholders) != 0:
                    word2id[token] = new_id
                    embeddings = np.concatenate((embeddings, np.random.randn(1, emb_size)))
                    new_token = placeholders.pop()
                    id2word[new_id] = new_token
                    ids[i] = new_id
                    tokens[i] = new_token
                    new_id += 1
            else:
                ids[i] = word2id[token]
                tokens[i] = id2word[ids[i]]

if __name__ == '__main__':

    with open(join("data","train","train_set.json"),'r') as f, \
        open(join("data","train","train_set_place.json"),'w') as f_write:
        f_list = f.readlines()
        for sample in f_list:
            dict_sample = json.loads(sample)
            answer_id = dict_sample['answer_sequence']
            answer_string = dict_sample['answer_string_sequence']
            question_id = dict_sample['question_sequence']
            question_string = dict_sample['question_string_sequence']
            document_id = dict_sample['document_sequence']
            document_string = dict_sample['string_sequence']

            substitute(document_id,document_string)
            substitute(answer_id,answer_string)
            substitute(question_id,question_string)
            text = json.dumps(dict_sample)
            f_write.write(text+'\n')

    save_path = join("data","dwr","place.glove.trimmed.{}.npz".format(emb_size))
    np.savez_compressed(save_path, glove=embeddings)

    cPickle.dump(word2id,open(join("data","train","place.word2id.pickle"),'w'),True)
    cPickle.dump(id2word,open(join("data","train","place.id2word.pickle"),'w'),True)


