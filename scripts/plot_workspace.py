import argparse
import pandas
import open3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--positions_file_1", type=str, required=True, 
                        help="Fingertips positions file.")

    parser.add_argument("-p2", "--positions_file_2", type=str, required=True, 
                    help="Fingertips positions file.")

    parser.add_argument("-p3", "--positions_file_3", type=str, required=True, 
                help="Fingertips positions file.")

    parser.add_argument("-p4", "--positions_file_4", type=str, required=True, 
            help="Fingertips positions file.")

    return parser.parse_args()


def read_points(positions_file):
    positions = pandas.read_csv(positions_file)
    pointcloud = open3d.geometry.PointCloud()

    points = open3d.utility.Vector3dVector(positions.to_numpy())
    pointcloud.points = points
    return pointcloud
    

def main():
    args = parse_args()

    pointcloud_1 = read_points(args.positions_file_1)
    pointcloud_1.paint_uniform_color([1.0, 0.0, 0.0])
    workspace_1 = pointcloud_1.compute_convex_hull()[0]
    workspace_1.paint_uniform_color([1.0, 0.0, 0.0])

    pointcloud_2 = read_points(args.positions_file_2)
    pointcloud_2.paint_uniform_color([0.5, 0.5, 0.0])
    workspace_2 = pointcloud_2.compute_convex_hull()[0]
    workspace_2.paint_uniform_color([0.5, 0.5, 0.0])

    pointcloud_3 = read_points(args.positions_file_3)
    pointcloud_3.paint_uniform_color([0.0, 1.0, 0.0])
    workspace_3 = pointcloud_3.compute_convex_hull()[0]
    workspace_3.paint_uniform_color([0.0, 1.0, 0.0])

    pointcloud_4 = read_points(args.positions_file_4)
    pointcloud_4.paint_uniform_color([0.0, 0.5, 0.5])
    workspace_4 = pointcloud_4.compute_convex_hull()[0]
    workspace_4.paint_uniform_color([0.0, 0.5, 0.5])

    origin = open3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.25)
    open3d.visualization.draw_geometries([workspace_1, workspace_2, workspace_3, workspace_4, origin])


if __name__ == '__main__':
    main()