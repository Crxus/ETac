from vedo import Mesh, show, Plotter
import numpy as np

mesh_path = "./new-leap.stl"  
mesh = Mesh(mesh_path).c("lightgray").pickable(True)

picked_points = []

plotter = Plotter(title="Pick point on surface")

def on_left_click(evt):
    if not evt.actor:
        return
    picked_point = evt.picked3d
    print(f"Picked 3D coordinate: {picked_point}")
    picked_points.append(picked_point)
    #plotter.add_point(picked_point, r=12, c='red')

plotter.add_callback("LeftButtonPress", on_left_click)

plotter.show(mesh, axes=1, interactive=True)

print("\nAll picked points:")
for i, p in enumerate(picked_points):
    print(f"{i+1}: {p}")

np.save('fuck.npy', picked_points)

'''
to read the points from the file
fuck = np.load('fuck.npy')  # or other file name 
'''
