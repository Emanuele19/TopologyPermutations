from abc import ABC, abstractmethod
import torch

class ModelInterface(torch.nn.Module, ABC):

    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, dropout:int, log=False):
        super().__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__hidden_channels = hidden_channels
        self.__dropout = dropout
        self.__log = log


    @property
    def in_channels(self):
        return self.__in_channels
    
    @property
    def out_channels(self):
        return self.__out_channels
    
    @property
    def hidden_channels(self):
        return self.__hidden_channels
    
    @property
    def dropout(self):
        return self.__dropout
    
    @property
    def log(self):
        return self.__log
    

class GraphModelInterface(ModelInterface):
    @abstractmethod
    def forward(self, x, edges_index):
        raise NotImplementedError()
    

class SimpleModelInterface(ModelInterface):
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()