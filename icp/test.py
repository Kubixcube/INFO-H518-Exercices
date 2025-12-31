import open3d as o3d

pcd = o3d.io.read_point_cloud("out_clouds_lr/cloud_right_cam.ply")
print(pcd)

o3d.visualization.draw_geometries([pcd])
