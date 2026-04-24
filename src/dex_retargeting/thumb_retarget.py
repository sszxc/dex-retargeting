import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# ==========================================
# 1. Strict parameter definitions
# ==========================================

# --- Base body (th_base) ---
BASE_POS = np.array([-0.0182, -0.019333, -0.045987])
BASE_QUAT_MJCF = [0.477714, 0.521334, -0.521334, 0.477714]
BASE_QUAT = [BASE_QUAT_MJCF[1], BASE_QUAT_MJCF[2], BASE_QUAT_MJCF[3], BASE_QUAT_MJCF[0]]

AXIS_0 = np.array([1, 0, 0])
LIMITS_0 = (0.263, 1.396)

OFFSET_1 = np.array([-0.027, -0.005, 0.0399])

# --- Joint 1 (thj1): negative Z in MJCF ---
AXIS_1 = np.array([0, 0, -1])
LIMITS_1 = (-0.105, 1.163)

OFFSET_2 = np.array([0, 0, 0.0177])

AXIS_2 = np.array([0, 1, 0])
LIMITS_2 = (-0.189, 1.644)

LOCAL_POINTING_VECTOR = np.array([0, 0, 1])

# ==========================================
# 2. Kinematics core
# ==========================================


def get_rotation_matrix(axis, theta):
    """Rodrigues rotation helper."""
    axis = axis / np.linalg.norm(axis)
    rot_vector = axis * theta
    r = R.from_rotvec(rot_vector)
    return r.as_matrix()


def forward_kinematics_direction(joint_angles):
    """End-effector direction in world frame for joint angles (th0, th1, th2)."""
    th0, th1, th2 = joint_angles

    r_base = R.from_quat(BASE_QUAT).as_matrix()

    r_j0 = get_rotation_matrix(AXIS_0, th0)
    R_0 = r_base @ r_j0

    r_j1 = get_rotation_matrix(AXIS_1, th1)
    R_1 = R_0 @ r_j1

    r_j2 = get_rotation_matrix(AXIS_2, th2)
    R_2 = R_1 @ r_j2

    v_world = R_2 @ LOCAL_POINTING_VECTOR

    return v_world / np.linalg.norm(v_world)

# ==========================================
# 3. IK solver
# ==========================================


def calculate_thumb_angles(target_direction_vector):
    """
    Args:
        target_direction_vector: (x, y, z) in world frame.

    Returns:
        (success, angles_rad, error_deg, actual_unit_vector)
    """
    target = np.array(target_direction_vector)
    norm = np.linalg.norm(target)
    if norm == 0:
        raise ValueError("Target direction must be non-zero")
    target = target / norm

    def objective(x):
        current_vec = forward_kinematics_direction(x)
        return -np.dot(current_vec, target)

    x0 = [
        np.mean(LIMITS_0),
        np.mean(LIMITS_1),
        np.mean(LIMITS_2),
    ]

    bounds = [LIMITS_0, LIMITS_1, LIMITS_2]

    res = minimize(objective, x0, bounds=bounds, method="SLSQP", tol=1e-6)

    final_vec = forward_kinematics_direction(res.x)
    dot_prod = np.dot(final_vec, target)
    error_rad = np.arccos(np.clip(dot_prod, -1.0, 1.0))
    error_deg = np.degrees(error_rad)

    return res.success, res.x, error_deg, final_vec

# ==========================================
# 4. Demo / verification
# ==========================================

if __name__ == "__main__":
    target_dir = [1, -1, 1]

    print(f"Target direction: {target_dir}")

    success, angles, error, actual_vec = calculate_thumb_angles(target_dir)

    print("\n" + "=" * 30)
    print("Results")
    print("=" * 30)

    if success:
        print("Optimizer: converged")
    else:
        print("Optimizer: did not converge — result may be unreliable")

    print("\nSuggested joint angles (rad):")
    print(f"  thj0 (Base):   {angles[0]:.4f}  [Range: {LIMITS_0}]")
    print(f"  thj1 (Prox):   {angles[1]:.4f}  [Range: {LIMITS_1}]")
    print(f"  thj2 (Medial): {angles[2]:.4f}  [Range: {LIMITS_2}]")

    print("\nGeometry check:")
    print(f"  Target (unit): {target_dir / np.linalg.norm(target_dir)}")
    print(f"  Actual (unit): {actual_vec}")
    print(f"  Angular error: {error:.2f}°")

    if error > 1.0:
        print("\n[Warning]")
        print("Large pointing error. Likely causes:")
        print("1. Target lies outside the reachable workspace.")
        print("2. Joint limits block that direction.")
        print("This is a physical limitation, not necessarily a bug; the solve is best-effort under constraints.")
