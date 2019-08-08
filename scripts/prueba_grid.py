import torch
import torch.nn.functional as F
import pdb
import numpy as np

A = torch.rand(1,1,8,5)
coords = np.array([(2,0), (2,2), (2,4), (2,6)]).astype(np.float32)
coords[:,0] = coords[:,0] / 2.0 -1
coords[:,1] = coords[:,1] / 3.5 -1
coords = coords.reshape(1, 4, 1, 2)
coords = torch.Tensor(coords)

result = F.grid_sample(A, coords)
pdb.set_trace()
print("FIN")
