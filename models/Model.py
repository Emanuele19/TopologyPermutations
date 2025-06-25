from abc import ABC, abstractmethod
import torch_geometric, torch

class ModelInterface(torch.nn.Module, ABC):

    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, dropout:int, log=False):
        super().__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__hidden_channels = hidden_channels
        self.__dropout = dropout
        self.__log = log

    @abstractmethod
    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """
        Performs a forward pass of the model.

        Parameters
            data : must contain the following fields
            - `x`: node features tensor of shape [num_nodes, num_features]

        Returns
            torch.Tensor: Output tensor
        """
        raise NotImplementedError
    
    @abstractmethod
    def train_model(self, data: torch_geometric.data.Data, optimizer, criterion, epochs:int, patience:int) -> None:
        """
        Performs training over `epochs` cycles.
        
        Parameters
            data: must contain the following fields
            - `x`: node features tensor of shape [num_nodes, num_features]

        Returns
            None
        """
        raise NotImplementedError


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
