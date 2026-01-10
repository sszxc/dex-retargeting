from __future__ import annotations

import time
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import rerun.blueprint as rrb
from rerun.blueprint.archetypes.force_link import ForceLink
from rerun.blueprint.archetypes.force_many_body import ForceManyBody
from rerun.blueprint.archetypes.force_position import ForcePosition
import numpy as np
from math import cos, sin, tau
from pathlib import Path
from typing import cast, List, Union
from uuid import uuid4
from utils.transform_utils import Transform
try:
    import trimesh
except ImportError:
    trimesh = None


class RerunBoard:
    def __init__(self, name, template=None):
        # assert name is not None, "name is required"
        rr.init(name, recording_id=uuid4(), spawn=True)
        # name example: f"CablePlug_{time.strftime('%m_%d_%H_%M', time.localtime())}"
        # use uuid4() to generate a unique recording_id 否则在同一个 process 中会重复

        if template == "3D":
            self.get_3D_view_board()
        elif template == "3D_image":
            self.get_3D_view_image_side_board()
        elif template == "3D_image_figure":
            self.get_3D_view_image_figure_side_board()
        elif template == "rrt":
            self.get_rrt_board()
        elif template == "image_image_figure":
            self.get_image_image_figure_board()
        else:
            # raise ValueError(f"template {template} not supported")
            pass

    def __getattr__(self, name):
        if hasattr(rr, name):
            return getattr(rr, name)  # 转发到 rerun 的方法
        else:
            raise AttributeError(f"'{name}' not found in rerun module.")

    def get_3D_view_board(self):
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(
                origin="/world",
                time_ranges=[
                    rrb.VisibleTimeRange(
                        timeline="simulation",
                        start=rrb.TimeRangeBoundary.cursor_relative(),
                        end=rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
            ),
            # collapse_panels=True,  # 最小化所有边栏
        )
        rr.send_blueprint(blueprint)

    def get_3D_view_image_side_board(self):
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/world",
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            timeline="simulation",
                            start=rrb.TimeRangeBoundary.cursor_relative(),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                ),
                rrb.Spatial2DView(
                    origin="/image",
                ),
            ),
            rrb.SelectionPanel(state="collapsed"),
        )
        rr.send_blueprint(blueprint)

    def get_image_image_figure_board(self):
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin="/graph",
                ),
                rrb.Spatial2DView(
                    origin="/image",
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        name="control_cost",
                        origin="/control_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="contact_cost",
                        origin="/contact_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="grasp_closure_cost",
                        origin="/grasp_closure_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="position_cost",
                        origin="/position_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="quaternion_cost",
                        origin="/quaternion_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="phy_cost",
                        origin="/phy_cost",
                    ),
                ),
                column_shares=[2, 2, 1],
            ),
            rrb.SelectionPanel(state="collapsed"),
        )
        rr.send_blueprint(blueprint)

    def get_3D_view_image_figure_side_board(self):
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/world",
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            timeline="simulation",
                            start=rrb.TimeRangeBoundary.cursor_relative(),
                            # start=rrb.TimeRangeBoundary.cursor_relative(-3),
                            # start=rrb.TimeRangeBoundary.infinite(),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                    background=rrb.components.BackgroundKind(2),
                ),
                rrb.Spatial2DView(
                    origin="/image",
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        name="control_cost",
                        origin="/control_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="contact_cost",
                        origin="/contact_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="velocity_cost",
                        origin="/velocity_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="position_cost",
                        origin="/position_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="quaternion_cost",
                        origin="/quaternion_cost",
                    ),
                    rrb.TimeSeriesView(
                        name="phy_cost",
                        origin="/phy_cost",
                    ),
                ),
                column_shares=[2, 2, 1],
            ),
            rrb.SelectionPanel(state="collapsed"),
        )
        rr.send_blueprint(blueprint)

    def get_rrt_board(self):
        blueprint = rrb.Blueprint(
            rrb.GraphView(
                origin="node_link",
                name="Node-link diagram",
                force_link=ForceLink(distance=200),
                force_many_body=ForceManyBody(strength=-200),
                force_position=ForcePosition(enabled=False),
            ),
        )
        rr.send_blueprint(blueprint)

    def step(self):
        if not hasattr(self, "time_tick"):
            self.time_tick = 0
        self.time_tick += 1
        rr.set_time("my_time", sequence=self.time_tick)

    @staticmethod
    def log_axes(
        translation,
        rotation=None,
        root="3d_points",
        name="",
        axis_size=0.1,
        label=None,
        only_z=False,
    ):  # axis_size=0.25
        if isinstance(translation, Transform):
            rotation = translation.rotation.as_matrix()
            translation = translation.translation
        assert rotation is not None, "rotation is required"
        assert len(translation) == 3, "pose_t should be a 3D vector"
        assert rotation.shape == (3, 3), "pose_R should be a 3x3 matrix"

        axis_x = rotation.dot([axis_size, 0, 0])  # 取第一列
        axis_y = rotation.dot([0, axis_size, 0])
        axis_z = rotation.dot([0, 0, axis_size])

        rr.log(
            f"{root}/{name}/point",
            rr.Points3D(
                positions=translation,
                colors=[[255, 0, 0]],
                radii=axis_size / 80,
                labels=label,
            ),
        )
        if not only_z:
            rr.log(
                f"{root}/{name}/arrow_x",
                rr.Arrows3D(origins=translation, vectors=axis_x, colors=[[255, 0, 0]]),
            )
            rr.log(
                f"{root}/{name}/arrow_y",
                rr.Arrows3D(origins=translation, vectors=axis_y, colors=[[0, 255, 0]]),
            )
        rr.log(
            f"{root}/{name}/arrow_z",
            rr.Arrows3D(origins=translation, vectors=axis_z, colors=[[0, 0, 255]]),
        )

    @staticmethod
    def log_obj(
        file_path: str | trimesh.Scene,
        obj_name: str,
        root: str = "world/",
        transform: np.ndarray | None = None,
    ):
        """
        Example Usage:
        for i in range(0, 5):
            RerunBoard.log_obj(Path('3D_model/indicator/Box_blue.glb'),
                    f'box_{i}',
                    transform=Transform(Rotation.from_rotvec([i*10, 0, 0], degrees=True), [i*40, 0, 0]).as_matrix())
        """

        def _load_file(path: Path) -> trimesh.Scene:
            """加载场景文件并返回一个 trimesh.Scene 对象
            典型格式 .jlb, .pbj
            所有支持格式 ↓
            > print(trimesh.available_formats())
            ['dict', 'dict64', 'json', 'msgpack', 'stl', 'stl_ascii', 'ply', 'obj', 'off', 'glb', 'gltf', 'xyz', 'zip', 'tar.bz2', 'tar.gz']
            """
            # print(f"loading scene {path}…")
            mesh = trimesh.load(path, force="scene")
            # 返回对象 mesh.graph.nodes: dict_keys(['world', 'Box.obj'])
            return cast(trimesh.Scene, mesh)

        def _log_scene(
            scene: trimesh.Scene, node: str, path: str | None = None
        ) -> None:
            """递归地记录场景中的每个节点及其变换矩阵"""
            if node is None:
                breakpoint()  # 存在情况: node 名为 None
                return
            path = path + "/" + node if path else node

            parent = scene.graph.transforms.parents.get(
                node
            )  # 获取 node 的父节点 (parents 是一个 dict，key 是任一 node，value 是 parent)
            children = scene.graph.transforms.children.get(
                node
            )  # 获取 node 的子节点 List

            node_data = scene.graph.get(
                frame_to=node, frame_from=parent
            )  # 获取指定父子之间相对位置
            if node_data:
                # Log the transform between this node and its direct parent (if it has one!).
                if parent:
                    world_from_mesh = node_data[0]
                    rr.log(  # 会影响后面的位置 (这个坐标轴并不会可视化出来，多次 log 好像可以连续改变位置)
                        path,  # 'world/Labtern' 'world/Lantern/LanternPole_Lantern'
                        rr.Transform3D(
                            translation=trimesh.transformations.translation_from_matrix(
                                world_from_mesh
                            ),
                            mat3x3=world_from_mesh[0:3, 0:3],
                        ),
                    )

                # Log this node's mesh, if it has one.
                mesh = cast(trimesh.Trimesh, scene.geometry.get(node_data[1]))
                # scene.geometry 是一个 dict，对应存储的多个 mesh（包括 vertics/faces）
                if mesh is not None:
                    vertex_colors = None
                    vertex_texcoords = None
                    albedo_factor = None
                    albedo_texture = None

                    try:
                        vertex_texcoords = mesh.visual.uv
                        # trimesh uses the OpenGL convention for UV coordinates, so we need to flip the V coordinate
                        # since Rerun uses the Vulkan/Metal/DX12/WebGPU convention.
                        vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]
                    except Exception:
                        pass

                    try:
                        albedo_texture = mesh.visual.material.baseColorTexture
                        if mesh.visual.material.baseColorTexture is None:
                            raise ValueError()
                    except Exception:
                        # Try vertex colors instead.
                        try:
                            colors = mesh.visual.to_color().vertex_colors
                            if len(colors) == 4:
                                # If trimesh gives us a single vertex color for the entire mesh, we can interpret that
                                # as an albedo factor for the whole primitive.
                                albedo_factor = np.array(colors)
                            else:
                                vertex_colors = colors
                        except Exception:
                            pass

                    rr.log(
                        path,
                        rr.Mesh3D(
                            vertex_positions=mesh.vertices,  # 顶点位置
                            vertex_colors=vertex_colors,  # 顶点颜色
                            vertex_normals=mesh.vertex_normals,  # 顶点法线 type: ignore[arg-type]
                            vertex_texcoords=vertex_texcoords,  # 顶点纹理坐标
                            albedo_texture=albedo_texture,  # 漫反射贴图
                            triangle_indices=mesh.faces,  # 三角形索引
                            albedo_factor=albedo_factor,  # 漫反射因子
                        ),
                    )  # Transform3D 和 Mesh3D 一组，对应一个实体

            if children:
                for child in children:
                    _log_scene(scene, child, path)
            return

        if isinstance(file_path, str):
            scene = _load_file(Path(file_path))
        else:
            scene = file_path
        obj_root = next(iter(scene.graph.nodes))
        # 应用变换矩阵
        if transform is None:
            transform = np.identity(4)  # default identity matrix
        rr.log(
            root + obj_name,
            rr.Transform3D(
                translation=trimesh.transformations.translation_from_matrix(transform),
                mat3x3=transform[0:3, 0:3],
            ),
        )
        _log_scene(scene, obj_root, root + obj_name)


if __name__ == "__main__":
    board = RerunBoard(f"RerunTest_{time.strftime('%m_%d_%H_%M', time.localtime())}")

    # board.set_surveillance_camera() # camera id, 重启后id会变

    for t in range(0, 50):
        board.step()

        # fig 1
        sin_of_t = sin(float(t) / 100.0) * 3
        cos_of_t = cos(float(t) / 100.0) * 3
        board.log("1d_scalar/depth", rr.Scalars(sin_of_t))
        board.log("1d_scalar/mode", rr.Scalars(cos_of_t))
        board.log("1d_scalar/force_z", rr.Scalars(cos_of_t + 1))

        # fig 2
        fx = sin(float(t) / 100.0) * 3 + np.random.randn()
        board.log("force/x", rr.Scalars(fx))
        fy = cos(float(t) / 100.0) * 3 + np.random.randn()
        board.log("force/y", rr.Scalars(fy))
        fz = cos(float(t + 1) / 100.0) * 3 + np.random.randn()
        board.log("force/z", rr.Scalars(fz))
        board.log("torque/x", rr.Scalars(fx))
        board.log("torque/y", rr.Scalars(fy))
        board.log("torque/z", rr.Scalars(fz))

        # 3D
        point_3d = np.random.uniform(-1, 1, 3)  # 生成一个随机的 3D 点
        point_3d = point_3d / np.linalg.norm(point_3d)
        board.log(
            "3d_points/pos",
            rr.Points3D(positions=[point_3d], colors=[[255, 0, 0]], radii=0.01),
        )  # , static=True
        board.log(
            "3d_points/force",
            rr.Arrows3D(origins=[0, 0, 0], vectors=[point_3d], colors=[[0, 255, 0]]),
        )  # , static=True
        board.log(
            "3d_points/pos_3d",
            rr.Transform3D(translation=point_3d, mat3x3=np.random.rand(3, 3)),
        )

        print(t)
        time.sleep(0.05)
