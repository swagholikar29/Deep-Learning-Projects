%matplotlib inline

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0 #Start of sentence
EOS_token = 1 #End of sentence


class Lang:
    def __init__(self, name):
        self.name = name #Adds the name of the the new name 
        self.word2index = {}  #Create new empty array for word to index dictionary
        self.word2count = {} #Create new empty for word to count for counting the number of words
        self.index2word = {0: "SOS", 1: "EOS"}  #Initializing
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '): #Splitting the sentence to individual words
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index: #If this word does not exist as an indexed word yet
            self.word2index[word] = self.n_words #Add this word to the list of words and index it as the 2ndd, 3rd, etc word
            self.word2count[word] = 1 #First time occurence of new word-> count = 1
            self.index2word[self.n_words] = word # Create cross mapping of index to word
            self.n_words += 1 #Now we have 1 more word in our dictionary
        else:
            self.word2count[word] += 1 # Add to the count of already existing word that has been found again

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    ) #Convert to Unicode to ASCII

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip()) #Normalize to ASCII after removing spaces at the beginning and end of the extracted word 
    s = re.sub(r"([.!?])", r" \1", s) #Remove .?!
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) #Remove non-letter, non-punctuation characters
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    pairs = [list(reversed(p)) for p in pairs]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[1].split(' ')) < MAX_LENGTH and \
        len(p[0].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', False)
print(random.choice(pairs))

!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
!ls -lat

#Create empty arrays for storing the words and the embeddings
vocab,embeddings = [],[]

#Open the glove embedding file and refer to it as 'fi'
#Read the file, Remove the extra elements after the word and split the sentences
with open('glove.6B.100d.txt','rt') as fi:
    full_content = fi.read().strip().split('\n') 

#
for i in range(len(full_content)):
    ind_word = full_content[i].split(' ')[0] 
    ind_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(ind_word)
    embeddings.append(ind_embeddings)

import numpy as np
vocab_npa = np.array(vocab) #Store the vocab vector as a numpy array
embs_npa = np.array(embeddings) #Store the embeddings vector as a numpy array
#insert '<pad>' and '<unk>' tokens at start of vocab_npa.
vocab_npa = np.insert(vocab_npa, 0, '<pad>')
vocab_npa = np.insert(vocab_npa, 1, '<unk>')
print(vocab_npa[:10])

#<pad> Padding if 10 words don't exist in the sentence and <unk> for unknown words
pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

#insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
print(embs_npa.shape) #The embeddings vector shape 400000+2 words in vocab, represented by a vector of 100 numbers combination

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, glove_emb): #Defining the Encoder RNN class
        super(EncoderRNN, self).__init__() #Create an object of Encoder Class
        self.hidden_size = hidden_size

        if glove_emb == False: #If not Glove Embeddings, use this as Encoder
          self.embedding = nn.Embedding(input_size, hidden_size)
          self.gru = nn.GRU(hidden_size, hidden_size)

        else: #Glove Embeddings use this as Encoder
          self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float(), freeze=True)
          self.gru = nn.GRU(hidden_size, hidden_size)
          
    def forward(self, input, hidden): #Forward Pass Model
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self): #Create Init Hidden layer
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module): #Decoder Class
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__() #Decoder Class 
        self.hidden_size = hidden_size #Hidden Size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence): #Defining the function to convert the indexes for words in the sentences
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence): #Defining the function to create the indexes for words in every sentence add EOS to it
#Return a torch tensor of the indexes for the whole language dataset
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair): #Defining the tensors for the word in each matching pair of the I/P and O/P languages
#Return Input Tensor, Target Sensor
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

print(len(pairs))
random.shuffle(pairs) 
X_train, X_test = train_test_split(pairs, test_size=0.20)
print(random.choice(X_train))
print(len(X_train))

teacher_forcing_ratio = 0.5 #Amount 

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, autoencoder = False, pretrained = False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_nevery

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(X_train))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        if autoencoder:
          target_tensor = training_pair[0]
        else:
          target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
#matplotlib.use('TkAgg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(X_test)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


#########SUPERVISED TRAINING#############

hidden_size = 128
encoder_v1 = EncoderRNN(input_lang.n_words, hidden_size, glove_emb = False).to(device)
attn_decoder_v1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device) #output_lang.n_words, dropout_p=0.1

%matplotlib inline
%matplotlib notebook

trainIters(encoder_v1, attn_decoder_v1, 75000, print_every=5000, autoencoder = False)


evaluateRandomly(encoder_v1, attn_decoder_v1)

%matplotlib inline
%matplotlib notebook

output_words, attentions = evaluate(
    encoder_v1, attn_decoder_v1, "i am a good boy")
plt.matshow(attentions.numpy())


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
%matplotlib notebook
plt.plot([1, 2, 3, 4])
plt.show()

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder_v1, attn_decoder_v1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("good boy")

evaluateAndShowAttention("good girl")

#GLoVe 6B Embedding
# https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76


#######PRETRAINING AS AUTOENCODER################

hidden_size = 128
encoder_v2 = EncoderRNN(input_lang.n_words, hidden_size, glove_emb = False).to(device)
attn_decoder_v2 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device) #output_lang.n_words, dropout_p=0.1
%matplotlib inline
%matplotlib notebook
trainIters(encoder_v2, attn_decoder_v2, 75000, print_every=5000, autoencoder = True)

evaluateRandomly(encoder_v2, attn_decoder_v2)

#Using Pretrained Embeddings for English in the Encoder
#GLoVe Embedding

hidden_size = 100
encoder_v3 = EncoderRNN(input_lang.n_words, hidden_size, glove_emb = True).to(device)
attn_decoder_v3 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device) #output_lang.n_words, dropout_p=0.1
%matplotlib inline
%matplotlib notebook
trainIters(encoder_v3, attn_decoder_v3, 75000, print_every=5000, autoencoder = False)

evaluateRandomly(encoder_v3, attn_decoder_v3)

