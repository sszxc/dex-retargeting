import mujoco
import mujoco.viewer
from pynput import keyboard

# MuJoCo viewer key callback
def mujoco_key_callback(keycode):
    if chr(keycode) == " ":
        print("MuJoCo: Space pressed in viewer")

# Global pynput keyboard listener
def on_press(key):
    if key == keyboard.Key.space:
        print("pynput: Space pressed globally")

# Start pynput listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Launch MuJoCo viewer
model = mujoco.MjModel.from_xml_path(
    "/mnt/1tb1/xuechao/MuJoCo-Asset-Pipeline/asset/scene/teleop_scene_left_028_skillet_lid/teleop_scene_left_028_skillet_lid.xml"
)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data, key_callback=mujoco_key_callback) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
