import torch as th
import numpy as np
from . import math_utils

class MotionOptimizer(th.nn.Module):
    def __init__(self, params, lr=1e-3, thres=1e-1):
        for k, v in params.items():
            if isinstance(v, th.Tensor):
                params[k] = th.nn.Parameter(v)
            else: 
                params[k] = th.nn.Parameter(th.tensor(v).to(th.float32))
        self.params = params
        self.optimizer = th.optim.Adam([v for v in self.params.values()], lr=1e-3)
        self.thres = thres
        self.current_loss = 1e10
    
    def step(self):
        self.optimizer.step()
        
    def is_converged(self):
        if self.current_loss > self.thres:
            return True
        else: 
            return False
        
    def cal_loss(self, loss_fn):
        # To be implemented by the subclass
        pass
    
    def forward(self, x, target):
        self.optimizer.zero_grad()
        loss = self.cal_loss(x, target)
        self.current_loss = loss
        loss.backward()
        self.optimizer.step()
        return loss

class Motion2DOptimizer(MotionOptimizer):
    def __init__(self, xt, device, lr=1e-3, thres=1e-1):
        self.device = device
        distance = 3
        params_to_opt = {'distance':distance,
                         'xt': xt}
        
        super().__init__(params_to_opt, lr)
        
    def cal_loss(self, motion_3d, target):
        # input: (B, J, 3, T)
        # target: (B, J, 2, T)
        projected_motion = []
        motion_3d = motion_3d.permute(0, 3, 1, 2)    # (B, T, J, 3)
        # Remove motion_path (xz)
        motion_3d[:, :, :, [0, 2]] -= motion_3d[:, :, 0:1, [0, 2]]
        
        for i in range(motion_3d.shape[0]):
            projected_motion.append(math_utils.perspective_projection_without_rotate(motion_3d=motion_3d[i], distance=self.params['distance']))
        projected_motion = th.stack(projected_motion, dim=0)    # (B, T, J, 2)
        if target.shape[0] != projected_motion.shape[0]:
            target = target.repeat_inter(projected_motion.shape[0], 1, 1, 1)
        
        target = th.ones_like(projected_motion)
        loss = th.nn.functional.mse_loss(projected_motion, target)
        return loss

    
    def forward(self, x, target):
        return super().forward(x, target)
    
    