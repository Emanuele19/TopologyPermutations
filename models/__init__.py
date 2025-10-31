from models.Model import ModelInterface, GraphModelInterface, SimpleModelInterface

from models.GCN import GCN
from models.GraphSAGE import GCN_Sage
from models.MLP import MLP
from models.APPNPNet import APPNPNet
from models.SAGEResidual import SAGEResidual1L

__all__ = [GCN, GCN_Sage, MLP, APPNPNet, SAGEResidual1L,
           ModelInterface,
           GraphModelInterface,
           SimpleModelInterface]