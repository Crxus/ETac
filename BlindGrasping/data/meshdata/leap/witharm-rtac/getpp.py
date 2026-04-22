import trimesh
import urdfpy
from urdfpy import URDF
import numpy as np
import os

# 加载 URDF
urdf_path = './pptac/urdf/pptac.urdf'
robot = URDF.load(urdf_path)

# 创建一个空的 trimesh.Scene 或直接合并 mesh
combined_mesh = None

for link in robot.links:
    for visual in link.visuals:
        if visual.geometry.mesh is not None:
            # 获取 mesh 文件路径
            mesh_path = visual.geometry.mesh.filename
            # 处理 package:// 路径
            if mesh_path.startswith('package://'):
                # 替换为实际路径（需配置 ROS_PACKAGE_PATH 或手动映射）
                mesh_path = mesh_path.replace('package://', '/your/catkin_ws/src/').split('/')[0] + '/' + '/'.join(mesh_path.split('/')[2:])
            
            # 加载 mesh
            try:
                mesh_part = trimesh.load(mesh_path)
                
                # 应用变换：先 visual.origin，再 link 的当前位姿（若在 home pose）
                transform = visual.origin
                # 如果需要全局位姿，还需乘上 joint 链的累积变换
                # 这里简化处理：仅用 visual.origin
                mesh_transformed = mesh_part.apply_transform(transform)
                
                # 合并
                if combined_mesh is None:
                    combined_mesh = mesh_transformed
                else:
                    combined_mesh += mesh_transformed
            except Exception as e:
                print(f"Failed to load {mesh_path}: {e}")

# 导出为 trimesh 支持的格式
if combined_mesh is not None:
    combined_mesh.export('PPtac_full.stl')      # 或 .obj, .glb, .ply
    print("Exported to PPtac_full.stl")