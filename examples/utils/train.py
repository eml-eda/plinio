import os
import numpy as np
import random
import torch

# seeding everything to maximize reproducibility
def set_seed(seed=23):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

