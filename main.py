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
    for i in range(len(document)):
        padded_document[i] = document[i] + [PAD_token]*(max_s_length-len(document[i]))

    padded_document = Variable(torch.LongTensor(padded_document))
    padded_document = padded_document.cuda() if args.use_cuda else padded_document
    question = Variable(torch.LongTensor(question))
    question = question.cuda() if args.use_cuda else question
    answer = Variable(torch.LongTensor(answer))
    answer = answer.cuda() if args.use_cuda else answer
    answer_length = len(answer)

    # encoder encode
    encoder_hidden = encoder(padded_document,question,answer,use_cuda)

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


def train_epoch(encoder_model, decoder_model, learning_rate=0.01,plot_every=100,print_every=100):

    start = time.time()
    encoder_optimizer = optim.SGD(encoder_model.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder_model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    plot_losses = []

    with open('./data/validation-0.json','r') as js_file:

        file_list = js_file.readlines()

        n_iters = len(file_list)

        for idx,sample in enumerate(file_list):

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


            loss = train(sentences,question,answer,encoder_model,decoder_model,
                  encoder_optimizer,decoder_optimizer,criterion,args.use_cuda,MAX_LENGTH)

            print_loss_total += loss
            plot_loss_total += loss

            if idx % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, idx / n_iters),
                                             idx, idx / n_iters * 100, print_loss_avg))

            if idx % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)

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

def process_glove(args, vocab_list, save_path, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not os.path.exists(save_path):
        glove_path = args.dwr_path
        if random_init:
            glove = np.random.randn(len(vocab_list), args.emb_size)
        else:
            glove = np.zeros((len(vocab_list), args.emb_size))
        found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
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

if __name__ == "__main__":

    args = parse_args()
    # TODO: reinitialize vocab later
    vocab_list = initialize_vocabulary()
    process_glove(args, vocab_list, args.emb_path)

    embeddings = load_glove_embeddings(args.emb_path)

    # create model
    encoder_model = EncoderModel(embeddings, args.bow_hidden_size, args.hidden_size)
    encoder_model = encoder_model.cuda() if args.use_cuda else encoder_model
    decoder_model = DecoderRNN(embeddings, args.emb_size, args.hidden_size, len(vocab_list))
    decoder_model = decoder_model.cuda() if args.use_cuda else decoder_model

    train_epoch(encoder_model,decoder_model,args.lr,50,50)