import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class BoW(nn.Module):
    def __init__(self,emb_size,hidden_size,embedding):
        super(BoW,self).__init__()

        # initilize embeddings

        # add layers
        self.layer_1 = nn.Linear(2*emb_size,hidden_size,False)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size,1,False)
        self.softmax = nn.Softmax()
        self.embedding = embedding

    def forward(self, document,question,hidden_size):
        '''

        :param document:

        sentences with ids
        e.g. [[id1,id2,id3,...id97],[id1,id2,...id56],...]

        :param quesion:

        sentence with ids
        e.g. [id1,id2,...]

        :return:
        '''
        question = question.expand(document.size(0),question.size(0))
        # TODO: calculate using sum/len
        bow_x = torch.mean(self.embedding(question),1)
        bow_s = torch.mean(self.embedding(document),1)

        h = torch.cat((bow_x,bow_s),1)
        hidden = self.relu(self.layer_1(h))
        logits = self.layer_2(hidden)
        outputs = self.softmax(logits)

        return outputs


class DocumentSummary(nn.Module):
    def __init__(self):
        super(DocumentSummary,self).__init__()

    def forward(self, probability,emb_document):
        probability = torch.unsqueeze(probability,-1)
        return torch.sum(probability*emb_document,0)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        word_lens = input.size(0)
        # we run this at once(over the whole input sequcnce)
        embedded = input.view(word_lens, 1, -1)
        outputs,hidden = self.gru(embedded,hidden)
        return outputs,hidden

    def initHidden(self,use_cuda):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result.cuda() if use_cuda else result

class DecoderRNN(nn.Module):
    def __init__(self, embeddings,input_size,hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))

        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        output = self.log_softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self,use_cuda):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result.cuda() if use_cuda else result

class EncoderModel(nn.Module):
    def __init__(self,embeddings,bow_hidden_size,hidden_size):
        super(EncoderModel, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))

        self.bow = BoW(embeddings.shape[1],bow_hidden_size,self.embedding)
        self.soft_attention = DocumentSummary()
        # input is embeddings dimension
        self.encoder = EncoderRNN(embeddings.shape[1],hidden_size)

    def forward(self, document,question,use_cuda):

        probability = self.bow(document,question,self.hidden_size)
        summary = self.soft_attention(probability,self.embedding(document))
        # the encoder input is question and document summary
        _,encoder_hidden = self.encoder(torch.cat([self.embedding(question),summary],0),self.encoder.initHidden(use_cuda))

        return encoder_hidden