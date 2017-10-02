import argparse
import json
import logging

import time
from numpy import random
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os

import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm

from model import EncoderModel, DecoderRNN

EOS_token = 1
SOS_token = 0
# according to document.vocab
PAD_token = 3

MAX_LENGTH = 10
teacher_forcing_ratio = 0.5

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--emb_size', type=int, default=100)
    parser.add_argument('--emb_path', type=str, default="")
    parser.add_argument('--dwr_path', type=str, default="")
    parser.add_argument('--bow_hidden_size',type=int,default=128)
    parser.add_argument('--hidden_size',type=int,default=200)
    parser.add_argument('--lr',type=float,default=0.01)

    args = parser.parse_args()
    embed_path = args.emb_path or join("data", "glove.trimmed.{}.npz".format(args.emb_size))
    args.emb_path = embed_path

    args.dwr_path = args.dwr_path or join("dwr", "glove.6B.{}d.txt".format(args.emb_size))

    return args

def initialize_vocabulary():
    with open('./data/document.vocab','r') as file:
        return [line.split('\t')[1] for line in file]

def train(document, question, answer, encoder, decoder, encoder_optimizer, decoder_optimizer,criterion,use_cuda,
          max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    # pad document
    # may use this later
    # padded_document = pad_packed_sequence(document,False,padding_value=PAD_token)
    max_s_length = max([len(s) for s in document])

    padded_document = np.zeros((len(document),max_s_length),dtype=np.int)
    mask_document = np.zeros((len(document),max_s_length),dtype=np.int)
    for i in range(len(document)):
        padded_document[i] = document[i] + [PAD_token]*(max_s_length-len(document[i]))
        mask_document[i,:len(document[i])] = np.ones(len(document[i]),dtype=int)

    padded_document = Variable(torch.LongTensor(padded_document))
    padded_document = padded_document.cuda() if args.use_cuda else padded_document

    mask_document = Variable(torch.FloatTensor(mask_document))
    mask_document = mask_document.cuda() if args.use_cuda else mask_document

    question = Variable(torch.LongTensor(question))
    question = question.cuda() if args.use_cuda else question
    answer = Variable(torch.LongTensor(answer))
    answer = answer.cuda() if args.use_cuda else answer
    answer_length = len(answer)

    # encoder encode
    encoder_hidden = encoder(padded_document,mask_document,question,use_cuda)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))

    decoder_input = decoder_input.cuda() if args.use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(answer_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, answer[di])
            decoder_input = answer[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(answer_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, answer[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / answer_length


def evaluate(document, question,encoder, id2word,decoder,use_cuda,max_length=MAX_LENGTH):

    # pad document
    # may use this later
    # padded_document = pad_packed_sequence(document,False,padding_value=PAD_token)
    max_s_length = max([len(s) for s in document])

    padded_document = np.zeros((len(document), max_s_length), dtype=np.int)
    mask_document = np.zeros((len(document), max_s_length), dtype=np.int)
    for i in range(len(document)):
        padded_document[i] = document[i] + [PAD_token] * (max_s_length - len(document[i]))
        mask_document[i, :len(document[i])] = np.ones(len(document[i]), dtype=int)

    padded_document = Variable(torch.LongTensor(padded_document))
    padded_document = padded_document.cuda() if args.use_cuda else padded_document

    mask_document = Variable(torch.FloatTensor(mask_document))
    mask_document = mask_document.cuda() if args.use_cuda else mask_document

    question = Variable(torch.LongTensor(question))
    question = question.cuda() if args.use_cuda else question

    # encoder encode
    encoder_hidden = encoder(padded_document, mask_document,question, use_cuda)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))

    decoder_input = decoder_input.cuda() if args.use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('</s>')
            break
        else:
            decoded_words.append(id2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words

def evaluateRandomly(encoder, id2word,decoder,use_cuda,num2eval=10):

    with open('./data/validation-1.json','r') as f:

        file_list = f.readlines()

        for _ in range(num2eval):

            sample = random.choice(file_list)
            dict_sample = json.loads(sample)

            document,sentences,question,answer = prepare_sample(sample)

            # skip illed sample
            if len(document) == 0 or len(answer) == 0 or len(question) == 0:
                continue

            # print info
            # TODO: change color of logger info
            logger.info("Document: {}".format(' '.join(dict_sample['string_sequence']).encode('utf8')))
            logger.info("Question: {}".format(' '.join(dict_sample['question_string_sequence']).encode('utf8')))
            logger.info("Real Answer: {}".format(' '.join(dict_sample['answer_string_sequence']).encode('utf8')))

            pred_answer = evaluate(sentences,question,encoder,id2word,decoder,use_cuda)
            logger.info("Predicted: {}".format(' '.join(pred_answer).encode('utf8')))

def train_epoch(encoder_model, decoder_model, learning_rate=0.01,plot_every=100,print_every=100):

    start = time.time()
    encoder_optimizer = optim.SGD(encoder_model.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder_model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    plot_losses = []
    illed_sample_num = 0

    with open('./data/validation-0.json','r') as js_file:

        file_list = js_file.readlines()

        n_iters = len(file_list)

        for idx,sample in enumerate(file_list,1):

            document,sentences,question,answer = prepare_sample(sample)

            # skip illed sample
            if len(document) == 0 or len(answer) == 0 or len(question) == 0:
                illed_sample_num += 1
                continue

            loss = train(sentences,question,answer,encoder_model,decoder_model,
                  encoder_optimizer,decoder_optimizer,criterion,args.use_cuda,MAX_LENGTH)

            print_loss_total += loss
            plot_loss_total += loss

            if idx % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, float(idx) / n_iters),
                                             idx, float(idx) / n_iters * 100, print_loss_avg))

            if idx % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)
        logger.info("Illed sample number: {}".format(illed_sample_num))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s):
    m = np.math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def process_glove(args, word2id, save_path, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not os.path.exists(save_path):
        glove_path = args.dwr_path
        if random_init:
            glove = np.random.randn(len(word2id), args.emb_size)
        else:
            glove = np.zeros((len(word2id), args.emb_size))
        found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh,total=5e5):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in word2id:
                    idx = word2id[word]
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in word2id:
                    idx = word2id[word.capitalize()]
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in word2id:
                    idx = word2id[word.upper()]
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))

def load_glove_embeddings(embed_path):
    logger.info("Loading glove embedding...")
    glove = np.load(embed_path)['glove']
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    return glove

def prepare_sample(sample):
    '''
    parse json sample to document answer and question
    :param json_line: sample line
    :return: document sentences question answer
    '''
    dict_sample = json.loads(sample)
    # use docuement vocab
    answer = dict_sample['answer_sequence']
    question = dict_sample['question_sequence']
    document = dict_sample['document_sequence']

    sentence_breaks = dict_sample['sentence_breaks']
    paragraph_breaks = dict_sample['paragraph_breaks']

    # TODO: may add paragraph breaks later
    breaks = sorted(sentence_breaks)
    breaks = [0] + breaks + [len(document)]

    sentences = []
    for i, id_break in enumerate(breaks[:-1]):
        start = breaks[i]
        end = breaks[i + 1]
        sentences.append(document[start:end])

    return document,sentences,question,answer

if __name__ == "__main__":

    args = parse_args()
    vocab_list = initialize_vocabulary()
    id2word = {idx:word for idx,word in  enumerate(vocab_list)}
    word2id = {word:idx for idx,word in  enumerate(vocab_list)}
    process_glove(args, word2id, args.emb_path)

    embeddings = load_glove_embeddings(args.emb_path)

    # create model
    encoder_model = EncoderModel(embeddings, args.bow_hidden_size, args.hidden_size)
    encoder_model = encoder_model.cuda() if args.use_cuda else encoder_model
    decoder_model = DecoderRNN(embeddings, args.emb_size, args.hidden_size, len(vocab_list))
    decoder_model = decoder_model.cuda() if args.use_cuda else decoder_model
    # TODO: batch input inplementation
    # TODO: placeholder inplementation
    train_epoch(encoder_model,decoder_model,args.lr,50,50)

    evaluateRandomly(encoder_model,id2word,decoder_model,args.use_cuda,num2eval=4)