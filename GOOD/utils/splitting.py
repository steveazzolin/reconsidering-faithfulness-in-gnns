import torch
from torch_geometric.utils import degree, cumsum, scatter, softmax

def split_graph(data, edge_score, ratio, debug=False, return_batch=False):
    # if debug:
    #     print("\nstarting")
    #     for i in range(6):
    #         print(data.edge_index[:, data.batch[data.edge_index[0]] == i])
    #         print(edge_score[data.batch[data.edge_index[0]] == i])
    #         sa,re = sparse_sort(
    #             edge_score[data.batch[data.edge_index[0]] == i],
    #             torch.zeros_like(edge_score[data.batch[data.edge_index[0]] == i], dtype=torch.long), descending=True)
    #         print(sa)
    #         print(re)
    #         print()
    #         print()

    has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None

    if data.batch is None:
        index = torch.zeros(data.num_nodes).to(torch.long)[data.edge_index[0]]
    else:
        index = data.batch[data.edge_index[0]]
    
    # new_idx_reserve, new_idx_drop, _, perm, mask = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True, debug=debug)
    new_idx_reserve, new_idx_drop, perm, mask = topK(edge_score, ratio, index, min_score=None, debug=debug)    
    
    if debug:
        index = data.batch[data.edge_index[0]]
        num_nodes = degree(index, dtype=torch.long)
        
        print(new_idx_reserve)
        print(num_nodes)
        print(data.batch.shape)
        print()
        new_batch = data.batch[new_causal_edge_index[0]]
        for i in range(6):
            print(data.edge_index[:, perm[index == i][mask[index == i]]])
            print(data.edge_index[:, perm[index == i][mask[index == i]]].unique())
            print(new_causal_edge_index[:, new_batch == i].unique())
            print()

    new_causal_edge_index = data.edge_index[:, new_idx_reserve]
    new_spu_edge_index = data.edge_index[:, new_idx_drop]

    new_causal_edge_weight = edge_score[new_idx_reserve]
    new_spu_edge_weight = - edge_score[new_idx_drop]

    if has_edge_attr:
        new_causal_edge_attr = data.edge_attr[new_idx_reserve]
        new_spu_edge_attr = data.edge_attr[new_idx_drop]
    else:
        new_causal_edge_attr = None
        new_spu_edge_attr = None

    if return_batch:
        causal_batch = data.batch[new_causal_edge_index[0]]
        mask_tmp = torch.zeros(data.edge_index.shape[1], device=data.edge_index.device, dtype=torch.bool)
        mask_tmp[new_idx_reserve] = 1
        return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight, causal_batch), \
            (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight), mask_tmp
    else:
        return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
            (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight)
    
def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    r'''
    Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
    '''
    f_src = src.double()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min) / (f_max - f_min + eps) + index.double() * (-1) ** int(descending)
    perm = norm.argsort(dim=dim, descending=descending)
    return src[perm], perm


def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12, debug=False):
    # print("\nsparse_topk")
    # print(index.unique(return_counts=True))
    rank, perm = sparse_sort(src, index, dim, descending, eps)
    num_nodes = degree(index, dtype=torch.long)
    # print(num_nodes)
    k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
    # print(k)
    start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
    mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
    mask = torch.cat(mask, dim=0)
    # print(mask)
    mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
    # print(mask)
    topk_perm = perm[mask] #topk edges selected
    exc_perm = perm[~mask]

    if debug:
        # print(src[index == 0])
        # print(src[index == 1])
        # print(src[index == 20])
        # print()
        # print(perm[index == 0])
        # print(perm[index == 1])
        # print(perm[index == 20])
        print()
        print(perm[index == 0][mask[index == 0]])
        print(perm[index == 1][mask[index == 1]])
        print(perm[index == 2][mask[index == 2]])
        print(perm[index == 3][mask[index == 3]])
        print(perm[index == 4][mask[index == 4]])
        print(perm[index == 5][mask[index == 5]])
        print(perm[index == 6][mask[index == 6]])
        # print(mask[index == 20])

    return topk_perm, exc_perm, rank, perm, mask

def relabel(x, edge_index, batch, pos=None):
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos

def topK(x, ratio, batch, min_score, tol=1e-7, debug=False):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio >= 1.:
        return torch.arange(x.shape[0], device=x.device), torch.tensor([], dtype=torch.long), None, None

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True, stable=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        # if debug:
        #     print(batch_perm)
        #     print(mask)
        #     exit()

        return x_perm[batch_perm[mask]], x_perm[batch_perm[~mask]].sort()[0], x_perm, mask

    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")