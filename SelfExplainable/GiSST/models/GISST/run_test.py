import torch
from torch_geometric.data import Data
from synthesize_graph import syn_ba_house, syn_node_feat
from sig.utils.synthetic_graph import build_house, build_cycle
from sig.utils.pyg_utils import get_pyg_edge_index
from sig.nn.loss.regularization_loss import reg_sig_loss
from sig.nn.loss.classification_loss import cross_entropy_loss
from sig.nn.models.sigcn import SIGCN
from sig.explainers.sig_explainer import SIGExplainer

graph, labels, _ = syn_ba_house()
node_feat, _ = syn_node_feat(labels, sigma_scale=0.1)
num_class = len(set(labels))

print('graph: {}'.format(type(graph)))
print('labels: {}'.format(type(labels)))
print('node_feat: {}'.format(type(node_feat)))

print('graph num nodes: {}'.format(graph.number_of_nodes()))
print('graph num edges: {}'.format(graph.number_of_edges()))
print('labels len: {}'.format(len(labels)))
print('node_feat shape: {}'.format(node_feat.shape))

data = Data()
data.x = torch.Tensor(node_feat)
data.y = torch.LongTensor(labels)
data.edge_index = get_pyg_edge_index(graph)

print(data)

model = SIGCN(
    input_size=data.x.size(1),
    output_size=num_class,
    hidden_conv_sizes=(8, 8), 
    hidden_dropout_probs=(0, 0)
)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.05, 
    weight_decay=0.001
)

def train():
    model.train()
    optimizer.zero_grad()
    out, x_prob, edge_prob = model(
        data.x, 
        data.edge_index, 
        return_probs=True
    )
    loss_x_l1, \
    loss_x_ent, \
    loss_edge_l1, \
    loss_edge_ent = reg_sig_loss(
        x_prob, 
        edge_prob, 
        coeffs={
            'x_l1': 0.01,
            'x_ent': 0.05,
            'edge_l1': 0.01,
            'edge_ent': 0.05
        }
    )
    
    loss = cross_entropy_loss(out, data.y) \
        + loss_x_l1 + loss_x_ent + loss_edge_l1 + loss_edge_ent
    loss.backward(retain_graph=True)
    optimizer.step()

for epoch in range(1, 51):
    train()
    if epoch % 10 == 0:
        print('Finished training for %d epochs' % epoch)
    
explainer = SIGExplainer(model)

node_feat_prob, edge_prob = explainer.explain_node(
    node_index=15,
    x=data.x,
    edge_index=data.edge_index,
    use_grad=False,
    y=None,
    loss_fn=None,
    take_abs=False,
    pred_for_grad=False
)

print('node_feat_prob shape: {}'.format(node_feat_prob.shape))
print(node_feat_prob)

print('edge_prob shape: {}'.format(edge_prob.shape))
print(edge_prob)

node_feat_score, edge_score = explainer.explain_node(
    node_index=15,
    x=data.x,
    edge_index=data.edge_index,
    use_grad=True,
    y=data.y,
    loss_fn=cross_entropy_loss,
    take_abs=False,
    pred_for_grad=True
)

print('node_feat_score shape: {}'.format(node_feat_score.shape))
print(node_feat_score)

print('edge_score shape: {}'.format(edge_score.shape))
print(edge_score)