################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
import torch.nn as nn
from torchvision import models


# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want

    raise NotImplementedError("Model Factory Not Implemented")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        last_layer = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*last_layer)
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