"""
从保存的 npz 读取 qpos 与场景路径，用 MuJoCo 加载对应 xml、设置 data.qpos 并离屏渲染一帧，保存图像。

- 若 xml 中无相机：用 MjvCamera 动态添加 3 个相机，均 look at (0, 0, 0.2)，多视角渲染。
- 若 xml 中有相机：渲染所有 xml 中的相机。
- 多张图上下拼接成一张图输出。

用法:
    python frame_render.py <path_to.npz> [--out image.png] [--height 720] [--width 1280]
    若 path 为文件夹，则迭代处理该目录下所有 .npz 文件，每个输出为同目录同名的 .png。
"""
from pathlib import Path

import argparse
import sys

import numpy as np
import mujoco
from tqdm import tqdm

# 无 xml 相机时使用的 3 个动态相机参数：均 look at (0, 0, 0.2)
DEFAULT_LOOKAT = [0.0, 0.0, 0.2]
DYNAMIC_CAMERAS = [
    {"distance": 0.8, "elevation": -20, "azimuth": 45},
    {"distance": 0.5, "elevation": -20, "azimuth": 90},
    {"distance": 0.8, "elevation": -20, "azimuth": 135},
]


def _make_dynamic_camera(par):
    """根据参数字典创建并配置一个 MjvCamera，look at DEFAULT_LOOKAT。"""
    camera = mujoco.MjvCamera()
    try:
        mujoco.mjv_defaultCamera(camera)
    except AttributeError:
        pass  # 部分版本无此函数，下面手动设置的参数已足够
    camera.lookat[:] = DEFAULT_LOOKAT
    camera.distance = par["distance"]
    camera.elevation = par["elevation"]
    camera.azimuth = par["azimuth"]
    return camera


def render_one(npz_path: Path, args, out_path: Path | None = None, *, verbose: bool = False) -> None:
    """渲染单个 npz 并保存到 out_path（None 时用 npz 同目录同名 .png）。verbose 为 True 时打印加载/渲染详情。"""
    if not npz_path.exists() or not npz_path.is_file():
        raise FileNotFoundError(f"npz 文件不存在: {npz_path}")

    data_npz = np.load(npz_path, allow_pickle=True)
    qpos = np.asarray(data_npz["qpos"]).copy()
    if verbose:
        print(f"[加载 npz] {npz_path}")
        print(f"[加载 qpos] shape={qpos.shape}, 具体数值:\n{qpos}")

    mj_xml_path_arr = data_npz["mj_xml_path"]
    mj_xml_path = str(mj_xml_path_arr.item() if mj_xml_path_arr.ndim == 0 else mj_xml_path_arr[()])

    xml_path = Path(mj_xml_path)
    if xml_path.is_dir():
        xml_path = next(xml_path.glob("*.xml"), None)
    if not xml_path or not Path(xml_path).exists():
        raise FileNotFoundError(f"场景 xml 不存在: {mj_xml_path}")

    if verbose:
        print(f"[加载 xml] 位置: {xml_path.resolve()}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if qpos.size != data.qpos.size:
        raise ValueError(
            f"npz 中 qpos 长度 {qpos.size} 与模型 nq {data.qpos.size} 不一致，请确认 npz 与 xml 对应"
        )
    data.qpos[:] = qpos
    if verbose:
        print(f"[设置 data.qpos] 已写入 {data.qpos.size} 维，与模型 nq 一致")
    mujoco.mj_forward(model, data)

    # 决定要渲染的相机列表：xml 有相机则用全部 xml 相机，否则用 3 个动态 MjvCamera
    if model.ncam > 0:
        cameras = list(range(model.ncam))  # 全部 xml 相机 id
    else:
        cameras = [_make_dynamic_camera(p) for p in DYNAMIC_CAMERAS]

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        frames = []
        for cam in cameras:
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            frames.append(pixels)
            if verbose:
                print(f"[渲染相机 {cam}] 已渲染")
        # 上下拼接：第一张在上，依次向下
        combined = np.vstack(frames)
    finally:
        renderer.close()

    if out_path is None:
        out_path = npz_path.with_suffix(".png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为 PNG（MuJoCo 返回 RGB）
    try:
        import imageio.v3 as iio
        iio.imwrite(out_path, combined)
    except Exception:
        try:
            import imageio as iio
            iio.imwrite(out_path, combined)
        except Exception:
            import cv2
            cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    if verbose:
        print(f"[拼接保存] 已渲染 {len(frames)} 张并上下拼接保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="从 npz 渲染一帧 MuJoCo 场景并保存图像")
    parser.add_argument(
        "npz_path",
        type=str,
        help="npz 文件或目录路径；为目录时迭代处理该目录下所有 .npz 文件",
    )
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="输出图像路径（仅单文件时有效），默认与 npz 同目录、同名 .npz 改为 .png")
    parser.add_argument("--height", type=int, default=480, help="每张子图渲染高度")
    parser.add_argument("--width", type=int, default=640, help="每张子图渲染宽度")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="打印加载/渲染详情（单文件默认开启）")
    parser.add_argument("--no-verbose", "--quiet", "-q", action="store_false", dest="verbose", help="关闭详情输出")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"路径不存在: {npz_path}")

    if npz_path.is_dir():
        npz_files = sorted(npz_path.glob("*.npz"))
        if not npz_files:
            print(f"目录下未找到 .npz 文件: {npz_path}", file=sys.stderr)
            sys.exit(1)
        for f in tqdm(npz_files, desc="渲染 npz"):
            try:
                render_one(f, args, out_path=None, verbose=False)
            except Exception as e:
                tqdm.write(f"处理 {f} 失败: {e}")
        return

    # 单文件：默认 verbose 开启，可用 -q/--no-verbose 关闭
    out_path = Path(args.out) if args.out is not None else None
    render_one(npz_path, args, out_path=out_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
