import yourdfpy
import trimesh
import os

def urdf_to_mesh_robust(urdf_path, output_path):
    print(f"正在加载 URDF: {urdf_path}")
    
    # 1. 使用 yourdfpy 加载
    # 它能自动处理 package:// 路径（前提是文件结构标准）
    # 如果你的 mesh 文件和 urdf 在同一个文件夹下，通常也能直接识别
    try:
        robot = yourdfpy.URDF.load(urdf_path)
    except ValueError as e:
        print("加载失败，请检查 URDF 文件中的 mesh 路径是否正确。")
        print(f"错误详情: {e}")
        return

    # 2. 获取 trimesh 场景对象
    # robot.scene 直接返回一个配置好坐标变换的 trimesh.Scene
    scene = robot.scene

    # 3. 将场景合并为一个单独的 Mesh
    # concatenate=True 会把所有零部件(Link)按当前姿态(默认是0位)合并
    print("正在合并几何体...")
    mesh = scene.dump(concatenate=True)

    # 4. 导出
    print(f"正在导出到: {output_path}")
    mesh.export(output_path)
    print("成功！")

if __name__ == "__main__":
    # 请替换为你的实际文件路径
    input_urdf = "./pptac.urdf" 
    output_file = "../PPTac_full.stl"
    
    # 检查文件是否存在
    if os.path.exists(input_urdf):
        urdf_to_mesh_robust(input_urdf, output_file)
    else:
        print(f"找不到文件: {input_urdf}")