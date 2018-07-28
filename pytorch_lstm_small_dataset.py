# objective is to make end 2 end system execute without fail
# import libraries
from keras.preprocessing import sequence
import pandas as pd
import pickle
import numpy as np
from normalization import normalize_documents
from utils import build_dataset
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import torch
from torch import nn
from torch.nn import LSTM

# defining parameters
vocabulary_size = 155  # len(vocabulary)
max_doc_len = 20
epoch = 500
seq_len = max_doc_len
input_size = vocabulary_size
batch_size = 2
hidden_size = 20
num_layers = 2
no_of_classes = 2



### defining dataset
dataset = {
    'quotes':["Where there is love there is life",
              "Happiness is when what you think what you say and what you do are in harmony"
              "Strength does not come from physical capacity It comes from an indomitable will"
              "In a gentle way you can shake the world"
              "The future depends on what we do in the present",
              "The weak can never forgive Forgiveness is the attribute of the strong",
              "A man is but the product of his thoughts; what he thinks, he becomes",
              "You must not lose faith in humanity Humanity is an ocean ",
              "Earth provides enough to satisfy every man's needs but not every man's greed",
              "Freedom is not worth having if it does not include the freedom to make mistakes",
              "Creativity is the greatest rebellion in existence",
              "Truth is not something outside to be discovered it is something inside to be realized",
              "The less people know the more stubbornly they know it",
              "Courage Is a Love Affair with the Unknown",
              "Life begins where fear ends",
              "Whatever you feel, you become It is your responsibility",
              "A certain darkness is needed to see the stars",
              "In love the other is important; in lust you are important",
              "One just needs a little alertness to see and find out: Life is really a great cosmic laughter",
              "Being is enlightenment becoming is ignorance",
              "In the space of no mind  truth descends like light",
              "Lovers never surrender to each other lovers simply surrender to love",
              "Only laughter makes a man rich, but the laughter has to be blissful"
              ],
    'author':[
              "gandhi","gandhi","gandhi","gandhi","gandhi","gandhi","gandhi","gandhi",
              "teresa", "teresa", "teresa", "teresa", "teresa", "teresa", "teresa", "teresa", "teresa", "teresa", "teresa", "teresa"
    ]
}

dataset = pd.DataFrame(dataset)
# load data # get the data - csv
# df = pd.read_csv("movie_reviews_small.csv", encoding="ISO-8859-1")

# prepare training and test data
# preparing test labels
y_train = np.array(dataset["author"])
no_classes = len(set(y_train))
y = np.array([int(item == 'gandhi') for item in y_train])
no_of_records = len(y)
# preparing test features
X = np.array(dataset["quotes"])
# normalized_documents = normalize_documents(X)

def doc_to_onehot(documents):
    vocabulary = set(' '.join(list(documents)).split(' '))
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump([dictionary, reverse_dictionary, data],handle)
    X_train = []
    for doc in list(documents):
       tmp = []
       for word in doc.split(' '):
           try:
               tmp.append(dictionary[word])
           except:
               pass
       X_train.append(tmp)

    X_train = sequence.pad_sequences(X_train, maxlen=max_doc_len)

    data = array(X_train)
    # one hot encode
    encoded = to_categorical(data)
    # invert encoding
    # inverted = argmax(encoded[0])
    return encoded

# x = np.zeros([batch_size, seq_len, input_size])
# y = np.zeros([batch_size])
x = doc_to_onehot(X)

# generate model simple RNN or LSTM
class RNN(nn.Module):
    # feed data to the network
    def __init__(self, input_size, num_layers, hidden_size, no_of_classes):
        super(RNN, self).__init__()
        self.no_of_classes = no_of_classes
        self.rnn = LSTM(input_size,hidden_size=hidden_size,num_layers=num_layers, batch_first=True, bias=True, bidirectional=False)
        self.fc1_in = nn.Linear(in_features=hidden_size*seq_len, out_features=hidden_size, bias=True)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=no_of_classes, bias=True)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        C0 = torch.zeros(num_layers, x.size(0), hidden_size)
        rnn_output, _ = self.rnn(x, (h0, C0))
        rnn_output = torch.reshape(rnn_output[:,:,:], (x.size(0),-1,))
        output = self.fc1_in(rnn_output)
        output = self.fc_out(output)
        return output

model = RNN(input_size, num_layers, hidden_size, no_of_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

loaded_model = RNN(input_size, num_layers, hidden_size, no_of_classes)
loaded_model.load_state_dict(torch.load('quotes.ckpt'))
loaded_model(torch.tensor(x[0:2]).float())

# print(model(torch.tensor(x[0:20]).float()))

for t in range(epoch):
    for i in range(int(no_of_records/batch_size)):
        x_batch = torch.tensor(x[i*batch_size:(i+1)*batch_size]).float()
        y_batch = torch.tensor(y[i*batch_size:(i+1)*batch_size]).long()
        y_pred = model(x_batch)

        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item())
        if (i + 1) % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(t + 1, epoch, i + 1, int(no_of_records/batch_size), loss.item()))

torch.save(model.state_dict(), 'quotes.ckpt')



# convert words in document to one hot representation
# prepare py-torch LSTM network
# feed data and get output - it should be binary
