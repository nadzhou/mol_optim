import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    # PYTHONHASHSEED only reaches child processes, never this interpreter. The real fix
    # for hash randomization is environment.valid_actions returning a sorted tuple rather
    # than a set; this line is belt-and-braces.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
