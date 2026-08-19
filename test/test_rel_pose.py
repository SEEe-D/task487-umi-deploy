import numpy as np

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-2)

def pose9d_to_mat(p9):
    T = np.eye(4)
    T[:3, :3] = rot6d_to_mat(p9[3:])
    T[:3, 3]  = p9[:3]
    return T

def mat_to_pose9d(T):
    return np.concatenate([T[:3, 3], T[:3, :2].reshape(-1)])  # 注意:rot6d 是 R 的前两行

def relative_pose(T_target, T_base):
    return np.linalg.inv(T_base) @ T_target


# # 
# Robot Right model: [ 0.06071195 -0.82923037 -0.8627812  -0.6020129  -0.28933449 -0.74422175
#  -0.0331165   0.94028594 -0.3387708 ]
# Robot Left model: [ 0.11226224 -0.68228907  0.0831547   0.23532034 -0.89853213 -0.37049203
#   0.6508715   0.42878644 -0.62650497]

right = np.array([0.06071195, -0.82923037, -0.8627812, -0.6020129, -0.28933449, -0.74422175, -0.0331165, 0.94028594, -0.3387708])
left  = np.array([0.11226224, -0.68228907, 0.0831547, 0.23532034, -0.89853213, -0.37049203, 0.6508715, 0.42878644, -0.62650497])

T_R = pose9d_to_mat(right)
T_L = pose9d_to_mat(left)

T_RwrtL = relative_pose(T_R, T_L)        # 右臂相对左臂
T_LwrtR = relative_pose(T_L, T_R)        # 左臂相对右臂(如果你想反过来)

print("Right wrt Left  (xyz+rot6d):", mat_to_pose9d(T_RwrtL))
print("Left  wrt Right (xyz+rot6d):", mat_to_pose9d(T_LwrtR))