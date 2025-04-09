r"""Initial process for fixing all possible random seed.
"""

import random

import numpy as np
import torch

from GOOD.utils.config_reader import Union, CommonArgs, Munch


def reset_random_seed(config: Union[CommonArgs, Munch], fixed_seed=None):
    r"""
    Initial process for fixing all possible random seed.

    Args:
       config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.random_seed`)


    """
    seed = config.random_seed if fixed_seed is None else fixed_seed
    # Fix Random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Default state is a training state
    torch.enable_grad()
