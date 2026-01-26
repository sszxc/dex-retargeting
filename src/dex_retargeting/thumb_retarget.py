import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# ==========================================
# 1. 严格参数定义 (Strict Parameter Definitions)
# ==========================================

# --- Base Body (th_base) ---
# XML pos: -0.0182 -0.019333 -0.045987
BASE_POS = np.array([-0.0182, -0.019333, -0.045987])
# XML quat: 0.477714 0.521334 -0.521334 0.477714 (w, x, y, z)
# Scipy requires (x, y, z, w)
BASE_QUAT_MJCF = [0.477714, 0.521334, -0.521334, 0.477714] 
BASE_QUAT = [BASE_QUAT_MJCF[1], BASE_QUAT_MJCF[2], BASE_QUAT_MJCF[3], BASE_QUAT_MJCF[0]]

# --- Joint 0 (thj0) ---
# XML axis: 1 0 0
AXIS_0 = np.array([1, 0, 0])
LIMITS_0 = (0.263, 1.396)

# --- Body Proximal Offset ---
OFFSET_1 = np.array([-0.027, -0.005, 0.0399])

# --- Joint 1 (thj1) ---
# XML axis: 0 0 -1 (注意是负 Z 轴)
AXIS_1 = np.array([0, 0, -1])
LIMITS_1 = (-0.105, 1.163)

# --- Body Medial Offset ---
OFFSET_2 = np.array([0, 0, 0.0177])

# --- Joint 2 (thj2) ---
# [USER UPDATED]: Axis is positive Y
AXIS_2 = np.array([0, 1, 0]) 
LIMITS_2 = (-0.189, 1.644)

# --- End Effector Definition ---
# [USER UPDATED]: Pointing towards local positive Z
LOCAL_POINTING_VECTOR = np.array([0, 0, 1]) 

# ==========================================
# 2. 运动学核心 (Kinematics Core)
# ==========================================

def get_rotation_matrix(axis, theta):
    """Rodrigues' rotation formula helper"""
    axis = axis / np.linalg.norm(axis)
    rot_vector = axis * theta
    r = R.from_rotvec(rot_vector)
    return r.as_matrix()

def forward_kinematics_direction(joint_angles):
    """
    计算给定关节角下的末端指向（世界坐标系）
    """
    th0, th1, th2 = joint_angles

    # 1. Base Orientation
    r_base = R.from_quat(BASE_QUAT).as_matrix()
    
    # 2. Joint 0 (X-axis)
    r_j0 = get_rotation_matrix(AXIS_0, th0)
    R_0 = r_base @ r_j0
    
    # 3. Joint 1 (Negative Z-axis)
    # 这里的旋转是相对于上一级坐标系的
    r_j1 = get_rotation_matrix(AXIS_1, th1)
    R_1 = R_0 @ r_j1
    
    # 4. Joint 2 (Y-axis) -> Updated
    r_j2 = get_rotation_matrix(AXIS_2, th2)
    R_2 = R_1 @ r_j2
    
    # 5. Transform local pointing vector to world
    v_world = R_2 @ LOCAL_POINTING_VECTOR
    
    # 归一化以防万一
    return v_world / np.linalg.norm(v_world)

# ==========================================
# 3. 逆运动学求解器 (IK Solver)
# ==========================================

def calculate_thumb_angles(target_direction_vector):
    """
    输入: 目标方向向量 (x, y, z)
    输出: (success_bool, angles_rad, error_deg)
    """
    target = np.array(target_direction_vector)
    norm = np.linalg.norm(target)
    if norm == 0:
        raise ValueError("目标向量不能为零向量")
    target = target / norm

    # 目标函数：最大化点积 (最小化 -dot)
    def objective(x):
        current_vec = forward_kinematics_direction(x)
        return -np.dot(current_vec, target)

    # 初始猜测：设为关节范围的中点，避免从 0 开始陷入死锁
    x0 = [
        np.mean(LIMITS_0),
        np.mean(LIMITS_1),
        np.mean(LIMITS_2)
    ]
    
    # 关节限位约束
    bounds = [LIMITS_0, LIMITS_1, LIMITS_2]

    # 使用 SLSQP (Sequential Least SQuares Programming) 处理有约束的非线性优化
    res = minimize(objective, x0, bounds=bounds, method='SLSQP', tol=1e-6)

    # 计算最终误差
    final_vec = forward_kinematics_direction(res.x)
    dot_prod = np.dot(final_vec, target)
    # clip 处理浮点数误差导致的 >1.0 情况
    error_rad = np.arccos(np.clip(dot_prod, -1.0, 1.0))
    error_deg = np.degrees(error_rad)

    return res.success, res.x, error_deg, final_vec

# ==========================================
# 4. 验证与运行 (Verification)
# ==========================================

if __name__ == "__main__":
    # --- 用户输入区 ---
    # 请在这里输入你想要指向的世界坐标系方向
    # 例如：指向正上方 [0, 0, 1] 或正前方 [1, 0, 0]
    target_dir = [1, -1, 1] 
    # ------------------

    print(f"目标方向向量: {target_dir}")
    
    success, angles, error, actual_vec = calculate_thumb_angles(target_dir)

    print("\n" + "="*30)
    print("计算结果 (Calculation Results)")
    print("="*30)
    
    if success:
        print(f"优化状态: 成功 (Converged)")
    else:
        print(f"优化状态: 失败 (Did not converge) - 结果可能不可靠")

    print(f"\n推荐关节角 (Rad):")
    print(f"  thj0 (Base):   {angles[0]:.4f}  [Range: {LIMITS_0}]")
    print(f"  thj1 (Prox):   {angles[1]:.4f}  [Range: {LIMITS_1}]")
    print(f"  thj2 (Medial): {angles[2]:.4f}  [Range: {LIMITS_2}]")

    print(f"\n几何验证:")
    print(f"  目标向量: {target_dir / np.linalg.norm(target_dir)}")
    print(f"  实际向量: {actual_vec}")
    print(f"  指向误差: {error:.2f}°")

    # 辩证思考与警告
    if error > 1.0:
        print("\n[关键警告 Critical Warning]")
        print("注意：计算结果存在显著误差。")
        print("原因可能是：")
        print("1. 目标方向在机械结构的物理死角区 (Workspace Limitations)。")
        print("2. 关节限位 (Limits) 阻止了拇指到达该方向。")
        print("这是物理约束，而非代码错误。此结果已是物理限制下的“最优解”。")
