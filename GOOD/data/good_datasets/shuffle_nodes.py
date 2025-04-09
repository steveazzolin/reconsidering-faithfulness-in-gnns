from typing import List, Optional, Union

from torch_geometric.data import Data, HeteroData

from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


class ShuffleGraph(BaseTransform):
    r"""Permutes nodes in the graph.

    Args:
        
    """
    def __init__(
        self
    ):
        pass

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        print(data)
        exit("here")
        for store in data.node_stores:
            if self.node_types is None or store._key in self.node_types:
                num_nodes = store.num_nodes
                assert num_nodes is not None
                c = torch.full((num_nodes, 1), self.value, dtype=torch.float)

                if hasattr(store, 'x') and self.cat:
                    x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                    store.x = torch.cat([x, c.to(x.device, x.dtype)], dim=-1)
                else:
                    store.x = c

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'