################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
from torchvision import models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        last_layer = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*last_layer)
        for params in self.resnet.parameters():
            params.require_grad = False
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.initialize_weights()
        
    def initialize_weights(self):
        self.embed.weight.data.normal_(0.0, 0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        resnet_outputs = self.resnet(images)
        resnet_outputs = Variable(resnet_outputs.data)
        resnet_outputs = resnet_outputs.view(resnet_outputs.size(0), -1)
        embedding = self.embed(resnet_outputs)
        return embedding

class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, model_type):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_type = model_type

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.vanilla_RNN = nn.RNN(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, nonlinearity = 'relu', batch_first = True)
        self.stacked_lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_weights(self.fc)

    def init_weights(self, m):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)

    def forward(self, features, captions):
        embedded_captions = self.embedding(captions)
        cat_captions = torch.cat((features.unsqueeze(1),embedded_captions), dim=1)
        if self.model_type == 'LSTM':
            hidden_outputs, _ = self.stacked_lstm(cat_captions)
        elif self.model_type == 'RNN':
            hidden_outputs, _ = self.vanilla_RNN(cat_captions)
        outputs = self.fc(hidden_outputs[:,:-1,:])
        return outputs

    def get_captions(self,features,mode,temperature,states=None):
        inputs = features.unsqueeze(1)
        word_ids = []
        for i in range(30):
            if self.model_type == 'LSTM':
                ht, states = self.stacked_lstm(inputs, states)
            elif self.model_type == 'RNN':
                ht, states = self.vanilla_RNN(inputs, states)
            output = self.fc(ht)
            if mode == "deterministic":
                predicted = output.argmax(2)
            elif mode == "stochastic":
                probs = F.softmax(output.div(temperature).squeeze(), dim=1)
                predicted = torch.multinomial(probs.data, 1)
            else:
                raise RuntimeError('Incorrect mode given.')
            word_ids.append(predicted)
            inputs = self.embedding(predicted)
        word_ids = torch.cat(word_ids, 1)
        return word_ids.squeeze()

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, model_type):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_type = model_type

        self.encoder = EncoderCNN(self.embed_size)
        self.decoder = RNNDecoder(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers, self.model_type)

    def forward(self,images,captions):
        features = self.encoder(images)
        outputs = self.decoder(features,captions)
        return outputs

    def generate_captions(self, images, mode, temperature=1):
        features = self.encoder(images)
        captions = self.decoder.get_captions(features, mode, temperature)
        return captions

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want
    model = EncoderDecoder(embedding_size, hidden_size, len(vocab), 2,model_type)
    return model