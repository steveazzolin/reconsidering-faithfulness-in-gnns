import torch.nn as nn
from models.ProtGNN.models.GCN import GCNNet, GCNNet_NC
from models.ProtGNN.models.GAT import GATNet, GATNet_NC
from models.ProtGNN.models.GIN import GINNet, GINNet_NC

__all__ = ['GnnNets', 'GnnNets_NC']


def get_model(input_dim, output_dim, model_args):
    if model_args.model_name.lower() == 'gcn':
        return GCNNet(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gat':
        return GATNet(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gin':
        return GINNet(input_dim, output_dim, model_args)
    else:
        raise NotImplementedError


def get_model_NC(input_dim, output_dim, model_args):
    if model_args.model_name.lower() == 'gcn':
        return GCNNet_NC(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gat':
        return GATNet_NC(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gin':
        return GINNet_NC(input_dim, output_dim, model_args)
    else:
        raise NotImplementedError


class GnnBase(nn.Module):
    def __init__(self):
        super(GnnBase, self).__init__()

    def forward(self, data):
        data = data.to(self.device)
        logits, prob, emb1, emb2, min_distances = self.model(data)
        return logits, prob, emb1, emb2, min_distances

    def update_state_dict(self, state_dict):
        original_state_dict = self.state_dict()
        loaded_state_dict = dict()
        for k, v in state_dict.items():
            if k in original_state_dict.keys():
                loaded_state_dict[k] = v
        self.load_state_dict(loaded_state_dict)

    def to_device(self):
        self.to(self.device)

    def save_state_dict(self):
        pass


class GnnNets(GnnBase):
    def __init__(self, input_dim, output_dim, model_args):
        super(GnnNets, self).__init__()
        self.model = get_model(input_dim, output_dim, model_args)
        self.device = model_args.device

    def forward(self, data, protgnn_plus=False, similarity=None):
        data = data.to(self.device)
        logits, prob, emb1, emb2, min_distances = self.model(data, protgnn_plus, similarity)
        return logits, prob, emb1, emb2, min_distances


class GnnNets_NC(GnnBase):
    def __init__(self, input_dim, output_dim, model_args):
        super(GnnNets_NC, self).__init__()
        self.model = get_model_NC(input_dim, output_dim, model_args)
        self.device = model_args.device

    def forward(self, data):
        data = data.to(self.device)
        logits, prob, emb, min_distances = self.model(data)
        return logits, prob, emb, min_distances
