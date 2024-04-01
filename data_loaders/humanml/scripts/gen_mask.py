import numpy as np
import torch as th

def gen_mask(mask_ratio, mask_idx, shape, num_joints=22):
    
    # Shape = [B, 263, 1, T]
    # Masking the features dimension
    jp = [2 + n*3 for n in range(num_joints-1)]   # local joint positions
    jp_offset = 4   # jp start at 4-idx (0-indexed) in the feature vector
    jp = np.array(jp) + jp_offset
    
    jv = [2 + n*3 for n in range(num_joints)]   # local joint velocities
    jv_offset = jp[-1] + 1   # jp[-1] is the last element in jp
    jv = np.array(jv) + jv_offset
    
    jr = [n for n in range(126)]   # local joint rotations
    jr_offset = jv[-1] + 1   # jr[-1] is the last element in jr
    jr = np.array(jr) + jr_offset
    
    # print(jp)
    # print(jv)
    # print(jr)
    
    # Masking
    # 1. root linear velocity on z-axis : idx=2
    # 2. root joint position on z-axis : idx=4, 5, ..., 64
    # 3. root joint velocity on z-axis
    # 4. local joint rotations (6D) : 
    f_to_mask = [2] + list(jp) + list(jv) + list(jr)
    f_mask = np.ones((1, shape[1], shape[2], shape[3]))
    f_mask[:, f_to_mask, :, :] = 0
    
    mask = np.ones(shape=shape)
    # print(f_mask.shape, mask.shape, mask_idx)
    
    if mask_ratio > 0:
        # Random the index to mask the features
        idx = np.arange(0, shape[0])
        # Remove the mask_idx from the idx
        idx = np.delete(idx, mask_idx)
        idx = np.random.choice(idx, int(mask_ratio * shape[0]), replace=False)
        mask_idx = np.concatenate((mask_idx, idx))
        
    # print(mask_idx)
    mask[mask_idx, :, :, :] = f_mask
    # print(mask.shape)
    # for m in range(mask.shape[0]):
    #     print(mask[m, :, :, 0:1].reshape(-1))
    #     print("=====================================")
        
    
    return mask
    
if __name__ == '__main__':
    shape = (10, 263, 1, 196)
    gen_mask(0.5, [1], shape, 22)