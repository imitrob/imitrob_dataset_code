#!/usr/bin/env python2

# This import registers the 3D projection, but is otherwise unused.
import copy
import errno

import rospy
import tf2_py
from geometry_msgs.msg import Point, TransformStamped
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tf.transformations as trans
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({'font.size': 22})

import rosbag
import trimesh
from tf_bag import BagTfTransformer
from graphviz import Digraph
from graphviz import Source
from sklearn import neighbors
import os
import shutil
import numpy as np
from scipy.spatial import distance, KDTree
from numba import jit


# bag_file_path = '/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/gluegun_trace_bag_2020-10-19-11-07-51.bag'
#
# bag = rosbag.Bag(bag_file_path)
#
# bag_transformer = BagTfTransformer(bag)
#
# dot = bag_transformer.getTransformGraphInfo()
# graph = Source(dot, 'tf_structure.gv', format='png')
# graph.view()
# Digraph.render(graph, view=True)

class ViveTraceAnalyzer(object):
    def __init__(self, tool_tracker='tracker_LHR_1D894210'):
        self.trackers = ['tracker_LHR_1D894210', 'tracker_LHR_786752BF']
        self.graph = None
        self.t_start = None
        self.folder = None
        self.bag_name = None

        self.tool_tracker = tool_tracker
        self.allpoints = np.array([])

    def load_from_bag(self, bag_file_path):
        bag = rosbag.Bag(bag_file_path)
        bag_transformer = BagTfTransformer(bag)

        # # draw tf graph
        # dot = bag_transformer.getTransformGraphInfo()
        # graph = Source(dot, 'tf_structure.gv', format='png')

        filename = os.path.basename(bag_file_path)
        self.folder = os.path.join(os.path.expanduser('~/trace_analysis/'), filename.split('.')[0])

        self.bag_name = filename.split('.')[0]

        try:
            os.makedirs(self.folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        dot = bag_transformer.getTransformGraphInfo()
        self.graph = Source(dot, os.path.join(self.folder, 'tf_structure.gv'), format='png')
        # graph.render()
        # graph.view()

        self.t_start = bag.get_start_time()

        pressed = []

        # t_start = None
        for key in bag.read_messages(topics=['/key']):
            # print(key.message)
            if key.message.data in 'pq':
                pressed.append((key.timestamp.to_sec() - self.t_start, key.message))

        pressed_p = [i[0] for i in pressed if i[1].data in 'pq']

        pressed_p = np.array(pressed_p)
        diffs = pressed_p[1:] - pressed_p[:-1]

        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.scatter(pressed_p, pressed_p.__len__() * [0.0])
        fig.savefig(os.path.join(self.folder, 'key_activation.png'))

        self.allpoints = np.array([])

        # for tf in bag.read_messages(topics=['/tf_static']):
        #     print(tf.message)
        # tracker2endpoint = np.eye(4)
        # tracker2endpoint[0,3] = 0.000823980795464
        # tracker2endpoint[1,3] = -0.00119503255084
        # tracker2endpoint[2,3] = 0.30897020896
        #
        # bag_transformer.populateTransformerAtTime()

        # calibrate the tool tracker... assumes that there are ONLY 2 trackers. The tool tracker is the one which moves
        # relative to the end_point
        idx1 = int(len(bag_transformer.tf_times)/4)
        t1, t2 = bag_transformer.tf_times[[idx1, idx1*2]]
        for tracker in self.trackers:
            translation1, quaternion1 = bag_transformer.lookupTransform(tracker, 'end_point',
                                                                      rospy.Time(nsecs=t1))
            translation2, quaternion2 = bag_transformer.lookupTransform(tracker, 'end_point',
                                                                        rospy.Time(nsecs=t2))
            if np.allclose(translation1, translation2):
                continue
            else:
                self.tool_tracker = tracker
                trackers = copy.copy(self.trackers)
                trackers.remove(self.tool_tracker)
                self.surface_marker = trackers[0]
                break



        # only consider transforms after the chain is complete
        last_time = bag_transformer.waitForTransform(self.tool_tracker, 'end_point', rospy.Time.from_sec(self.t_start)).to_sec()
        first_time = last_time

        # last_time = self.t_start - 1.0  # set time earlier than start time to record also first point
        for tf in bag.read_messages(topics=['/tf']):
            # print(tf)
            if tf.timestamp.to_sec() - last_time < 1/60.0:
                continue
            else:
                last_time = tf.timestamp.to_sec()

            if np.min(np.abs(pressed_p - (tf.timestamp.to_sec() - self.t_start))) > 0.02:
                continue

            try:
                translation, quaternion = bag_transformer.lookupTransform(self.tool_tracker, 'end_point',
                                                                          tf.timestamp)
            except tf2_py.ExtrapolationException as e:
                continue
            # translation, quaternion = bag_transformer.lookupTransform(self.tool_tracker, self.surface_marker,
            #                                                           tf.timestamp)

            # from tf2_geometry_msgs.tf2_geometry_msgs import PointStamped, do_transform_point
            # point = do_transform_point(PointStamped(point=Point(x=translation[0], y=translation[1], z=translation[2])),
            #                            TransformStamped(translation))

            if self.allpoints.__len__() == 0:
                self.allpoints = np.array(translation)
            else:
                # skip identical points
                if not np.allclose(translation, self.allpoints[-1]):
                    self.allpoints = np.vstack((self.allpoints, translation))

        self.tree = KDTree(self.allpoints, leafsize=10)

    def draw_points(self, neighbors=50, dist_median_threshold=0.01):
        allpoints_plus = np.copy(self.allpoints)
        allpoints_plus = np.hstack([allpoints_plus, np.zeros((allpoints_plus.shape[0], 1))])

        for pt in allpoints_plus:
            dist, ind = self.tree.query(pt[:-1], k=neighbors)
            pt[-1] = np.median(dist)

        fig = plt.figure()  # type: plt.Figure
        ax = fig.add_subplot(111, projection='3d')


        markersize = [2 for n in range(len(self.allpoints[:, 0]))]

        c = (allpoints_plus[:, 3] > 0.01)
        c2 = allpoints_plus[:, 3] / np.median(allpoints_plus[:, 3])
        ax.scatter(self.allpoints[:, 0], self.allpoints[:, 1], self.allpoints[:, 2], marker='x', s=markersize, c=c, cmap='coolwarm')
        ax.set_xlabel('x [m]', labelpad=15)
        ax.set_ylabel('y [m]', labelpad=15)
        ax.set_zlabel('z [m]', labelpad=15)
        ax.axis('equal')
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.folder, 'points_plot.png'))

    def draw_points_density(self, neighbors=50, dist_median_threshold=0.01, density_threshold=5, plot_density=True, markersize=20, draw_roller_bb=True, debug_prints=False):
        allpoints_plus = np.copy(self.allpoints)
        allpoints_plus = np.hstack([allpoints_plus, np.ones((allpoints_plus.shape[0], 1))])

        # for pt in allpoints_plus:
        #     dist, ind = self.tree.query(pt[:-1], k=neighbors)
        #     pt[-1] = np.median(dist)

        fig = plt.figure()  # type: plt.Figure
        ax = fig.add_subplot(111, projection='3d')


        markersize_arr = [markersize for n in range(len(self.allpoints[:, 0]))]

        d = np.zeros((allpoints_plus.shape[0], 1))
        for idx, p in enumerate(allpoints_plus[:]):
            c3 = (p - self.transform[0:4,3]) * 200
            int_array = np.rint(c3).astype(int)
            # c3 = np.linalg.inv(self.transform) * p  # self.density
            if debug_prints:
                print(c3)
            d[idx,0] = self.density[int_array[0],int_array[1],int_array[2]]

        if debug_prints:
            print(d)

        c = (d[:,0] > density_threshold)
        filtered = (d[:,0] <= density_threshold)
        filtered_points = self.allpoints[filtered]
        filtered_points_remain = self.allpoints[np.logical_not(filtered)]
        # c = (allpoints_plus[:, 3] > 0.01)
        # c3 = allpoints_plus[:].T* self.transform  #self.density
        # c2 = allpoints_plus[:, 3] / np.median(allpoints_plus[:, 3])
        if plot_density:
            cm = d[np.logical_not(filtered)]
            cm = cm[:,0] *0.01
            ax.scatter(filtered_points_remain[:, 0], filtered_points_remain[:, 1], filtered_points_remain[:, 2], marker='x', s=markersize, c=cm,
                       cmap='coolwarm')
            ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], marker='x', s=markersize, c='y')
        else:
            ax.scatter(self.allpoints[:, 0], self.allpoints[:, 1], self.allpoints[:, 2], marker='x', s=markersize_arr, c=c, cmap='coolwarm')
        if draw_roller_bb:
            bb_points = np.loadtxt('../data/roller_trace_bag_2020-12-09-15-04-33_bb_corners_z_aligned.csv',
            delimiter=',')
            Z = bb_points
            ax.scatter(bb_points[:, 0], bb_points[:, 1], bb_points[:, 2],
                       s=30, c='k')
            # verts = [[Z[0], Z[1], Z[2], Z[3]],
            #          [Z[4], Z[5], Z[6], Z[7]],
            #          [Z[0], Z[1], Z[5], Z[4]],
            #          [Z[2], Z[3], Z[7], Z[6]],
            #          [Z[1], Z[2], Z[6], Z[5]],
            #          [Z[4], Z[7], Z[3], Z[0]]]

            verts = [[Z[0], Z[1], Z[3], Z[2]],
                     [Z[4], Z[5], Z[7], Z[6]],
                     [Z[0], Z[1], Z[5], Z[4]],
                     [Z[2], Z[3], Z[7], Z[6]],
                     [Z[1], Z[5], Z[7], Z[3]],
                     [Z[4], Z[6], Z[2], Z[0]]]

            # plot sides
            faces = Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1)
            faces.set_facecolor((0, 0, 1, 0.1))
            ax.add_collection3d(faces)

#             bb_points = np.array(-1.295665069354463295e-01,-6.338174171493134168e-02,6.250000000000000000e-02
# -1.295665069354463295e-01,-6.338174171493134168e-02,-3.750000000000000555e-02
# -4.204223408581715671e-02,1.428125235448712560e-01,6.250000000000000000e-02
# -4.204223408581715671e-02,1.428125235448712560e-01,-3.750000000000000555e-02
# 1.552041901477959063e-01,-1.842597310212216177e-01,6.250000000000000000e-02
# 1.552041901477959063e-01,-1.842597310212216177e-01,-3.750000000000000555e-02
# 2.427284629974250652e-01,2.193453423858098342e-02,6.250000000000000000e-02
# 2.427284629974250652e-01,2.193453423858098342e-02,-3.750000000000000555e-02)
        ax.quiver([0], [0], [0], [0], [0], [0], length=0.1, normalize=True)
        ax.set_xlabel('x [m]', labelpad=15)
        ax.set_ylabel('y [m]', labelpad=15)
        ax.set_zlabel('z [m]', labelpad=15)
        # ax.axis('equal')
        ax.axis('auto')
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.folder, 'points_plot_density.png'))

    def calc_point_density(self, resolution=0.005,  kernel_size=0.01):
        # point density calculation

        # find grid borders

        x_range = [np.min(self.allpoints[:, 0]), np.max(self.allpoints[:, 0])]
        y_range = [np.min(self.allpoints[:, 1]), np.max(self.allpoints[:, 1])]
        z_range = [np.min(self.allpoints[:, 2]), np.max(self.allpoints[:, 2])]

        dx = np.floor(x_range[1] / resolution) - np.ceil(x_range[0] / resolution) + 3
        dy = np.floor(y_range[1] / resolution) - np.ceil(y_range[0] / resolution) + 3
        dz = np.floor(z_range[1] / resolution) - np.ceil(z_range[0] / resolution) + 3

        x_grid = np.linspace(np.floor(x_range[0] / resolution) * resolution,
                             np.ceil(x_range[1] / resolution) * resolution,
                             int(dx))
        y_grid = np.linspace(np.floor(y_range[0] / resolution) * resolution,
                             np.ceil(y_range[1] / resolution) * resolution,
                             int(dy))
        z_grid = np.linspace(np.floor(z_range[0] / resolution) * resolution,
                             np.ceil(z_range[1] / resolution) * resolution,
                             int(dz))

        self.density = np.zeros((int(dx), int(dy), int(dz)))

        @jit
        def calc_densities(density, x_grid, y_grid, z_grid, kernel_size):
            for index, x in np.ndenumerate(density):
                dist, ind = self.tree.query([x_grid[index[0]],
                                             y_grid[index[1]],
                                             z_grid[index[2]]],
                                             k=100,
                                             distance_upper_bound=kernel_size*1.1
                                             )
                density[index] = np.sum(dist <= kernel_size)
            return density

        self.density = calc_densities(self.density, x_grid, y_grid, z_grid, kernel_size)

        self.transform = np.eye(4)
        self.transform = self.transform * resolution
        self.transform[3, 3] = 1.0
        self.transform[0, 3] = x_grid[0]
        self.transform[1, 3] = y_grid[0]
        self.transform[2, 3] = z_grid[0]

    def plot_voxels(self):
        voxels = (self.density >= 18)

        # set the colors of each object
        colors = np.empty(voxels.shape, dtype=object)
        colors[voxels] = 'red'
        # colors[link] = 'red'
        # colors[cube1] = 'blue'
        # colors[cube2] = 'green'

        # and plot everything
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')

        fig.savefig(os.path.join(self.folder, 'voxel_plot.pdf'))

    def show_bb(self, density_threshold=18):
        voxels = (self.density >= density_threshold)
        voxelgrid = trimesh.voxel.VoxelGrid(voxels, transform=self.transform)
        # voxelgrid.show()

        mesh = voxelgrid.as_boxes()

        bb = mesh.bounding_box_oriented
        assert isinstance(bb, trimesh.primitives.Box)
        bb.visual.main_color[3] = 150
        bb.visual.face_colors[:, 3] = 150
        scene = (mesh + bb).show()
        # scene.save_image()
        scene = mesh.scene()
        scene.add_geometry(bb)
        # self.render_views(scene, self.folder, name='obb')

        # self.render_mesh(scene, path=os.path.join(self.folder, 'voxel_mesh_bb.png'))

        # mesh.vertices
        print(bb.center_mass)

        np.savetxt(os.path.join(self.folder, 'bb_corners.csv'), bb.vertices, delimiter=',')

        return bb.vertices

    # def save_image(self, mesh):
    #     pass

    def optimize_bb(self, density_threshold=18, render=False, ret_scene=False):
        voxels = (self.density >= density_threshold)
        voxelgrid = trimesh.voxel.VoxelGrid(voxels, transform=self.transform)
        # voxelgrid.show()

        from scipy.optimize import fmin

        def rotate_to_min(angle):
            trans = trimesh.transformations.euler_matrix(0, 0, angle)
            optimesh = voxelgrid.copy()
            volume = optimesh.apply_transform(trans).as_boxes().bounding_box.volume
            return volume * 10000000

        # res = fmin(rotate_to_min, x0=np.array([0.0]), full_output=True)
        res = fmin(rotate_to_min, x0=0.0, ftol=0.0000001, xtol=0.0000001, full_output=True)

        trans = trimesh.transformations.euler_matrix(0, 0, res[0][0])
        optimesh = voxelgrid.copy()
        optimesh = optimesh.as_boxes()
        optimesh.apply_transform(trans)

        bb = optimesh.bounding_box
        bb.visual.main_color[3] = 150
        bb.visual.face_colors[:, 3] = 150
        scene = (optimesh + bb).show()

        np.savetxt(os.path.join(self.folder, self.bag_name + '_bb_corners_z_aligned.csv'), bb.vertices, delimiter=',')

        scene = optimesh.scene()
        scene.add_geometry(bb)
        if render:
            self.render_views(scene, self.folder, name='optibox')



        print(res)
        # mesh = voxelgrid.as_boxes()
        #
        # bb = mesh.bounding_box_oriented
        # assert isinstance(bb, trimesh.primitives.Box)
        # bb.visual.main_color[3] = 150
        # bb.visual.face_colors[:, 3] = 150
        # scene = (mesh + bb).show()
        # # scene.save_image()
        #
        # # self.render_mesh(scene, path=os.path.join(self.folder, 'voxel_mesh_bb.png'))
        #
        # # mesh.vertices
        # print(bb.center_mass)

        # np.savetxt(os.path.join(self.folder, 'bb_corners.csv'), bb.vertices, delimiter=',')
        if ret_scene:
            return bb.vertices, scene
        return bb.vertices

    def render_mesh(self, scene, path=None):
        data = scene.save_image(resolution=(1024, 768), visible=True)
        # if this doesn't work, try it with a visible window:
        # data = scene.save_image(visible=True)

        if path is None:
            return data

        from PIL import Image
        rendered = Image.open(trimesh.util.wrap_as_stream(data))
        file = open(path, 'w+')
        # file.write(data)
        file.write(rendered)
        file.close()

    # import numpy as np
    # import trimesh
    #
    # if __name__ == '__main__':
    #     # print logged messages
    #     trimesh.util.attach_to_log()

    def select_bb(self, density_threshold=18):
        voxels = (self.density >= density_threshold)
        voxelgrid = trimesh.voxel.VoxelGrid(voxels, transform=self.transform)
        # voxelgrid.show()

        #
        # from scipy.optimize import fmin
        #
        # def rotate_to_min(angle):
        #     trans = trimesh.transformations.euler_matrix(0, 0, angle)
        #     optimesh = voxelgrid.copy()
        #     volume = optimesh.apply_transform(trans).bounding_box.volume
        #     return volume
        #
        # res = fmin(rotate_to_min, x0=np.array([0.0]), full_output=True)
        angles = np.linspace(0, np.pi * 0.5, 90)
        angles = np.array([0.5*np.pi * 23.0/90.0])
        for idx, angle in enumerate(angles):
            trans = trimesh.transformations.euler_matrix(0, 0, angle)
            optimesh = voxelgrid.copy()
            optimesh = optimesh.as_boxes()
            optimesh.apply_transform(trans)

            bb = optimesh.bounding_box  # type: trimesh.primitives.Box
            bb.visual.main_color[3] = 150
            bb.visual.face_colors[:, 3] = 150
            scene = (optimesh + bb).show()

            # np.savetxt(os.path.join(self.folder, self.bag_name + "_bb_corners_z_aligned_{:0>2d}.csv".format(idx)),
            #            bb.vertices,
            #            delimiter=',')

            # transform the bb vertices back to the original frame
            trans_inv = np.linalg.inv(trans)

            bb_mesh = bb.copy()
            bb_mesh_2 = trimesh.Trimesh(vertices=bb_mesh.vertices, faces=bb_mesh.faces)
            bb_mesh_2.apply_transform(trans_inv)
            np.savetxt(os.path.join(self.folder, self.bag_name + "_bb_corners_z_aligned_{:0>2d}.csv".format(idx)),
                       bb_mesh_2.vertices,
                       delimiter=',')


            scene = optimesh.scene()
            scene.add_geometry(bb)
        # self.render_views(scene, self.folder, name='optibox')

    def save_image(self, scene, folder, file_name):

        png = scene.save_image(resolution=[1920, 1080], visible=True)
        with open(os.path.join(folder, file_name), 'wb') as f:
            f.write(png)

    def render_views(self, mesh, folder, name=None):
        # load a mesh
        # mesh = trimesh.load('../models/featuretype.STL')

        # get a scene object containing the mesh, this is equivalent to:
        # scene = trimesh.scene.Scene(mesh)
        if isinstance(mesh, trimesh.Trimesh):
            scene = mesh.scene()
        elif isinstance(mesh, trimesh.Scene):
            scene = mesh
        # a 45 degree homogeneous rotation matrix around
        # the Y axis at the scene centroid
        rotate_z = trimesh.transformations.rotation_matrix(
            angle=np.radians(10.0),
            direction=[0, 0, 1],
            point=scene.centroid)

        rotate_x = trimesh.transformations.rotation_matrix(
            angle=np.radians(90.0),
            direction=[1, 0, 0],
            point=scene.centroid)

        rotate_y = trimesh.transformations.rotation_matrix(
            angle=np.radians(10.0),
            direction=[0, 1, 0],
            point=scene.centroid)

        # rotate the camera view transform
        camera_old, _geometry = scene.graph[scene.camera.name]
        camera_new = np.dot(rotate_x, camera_old)

        # apply the new transform
        scene.graph[scene.camera.name] = camera_new


        for k in range(18):
            # rotate the camera view transform
            camera_old, _geometry = scene.graph[scene.camera.name]
            camera_new = np.dot(rotate_y, camera_old)

            # apply the new transform
            scene.graph[scene.camera.name] = camera_new

            for i in range(36):
                trimesh.constants.log.info('Saving image %d', i)

                # rotate the camera view transform
                camera_old, _geometry = scene.graph[scene.camera.name]
                camera_new = np.dot(rotate_z, camera_old)

                # apply the new transform
                scene.graph[scene.camera.name] = camera_new

                # saving an image requires an opengl context, so if -nw
                # is passed don't save the image
                try:
                    # increment the file name
                    if name is None:
                        file_name = 'render_' + str(i) + '.png'
                    else:
                        file_name = 'render_' + name + '_' + str(k) + '_' + str(i) + '.png'
                    # save a render of the object as a png
                    self.save_image(scene, folder, file_name)
                    # png = scene.save_image(resolution=[1920, 1080], visible=True)
                    # with open(os.path.join(folder, file_name), 'wb') as f:
                    #     f.write(png)


                except BaseException as E:
                    print("unable to save image", str(E))




# if __name__ == "__main__":
#     bags = ['/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/gluegun_trace_bag_2020-09-22-11-39-19.bag'#,
#             # '/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/hammer_trace_2020-09-18-17-25-43.bag',
#             # '/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/hladitko_trace_bag_2020-09-22-16-02-38.bag.active'] #,
#             # '/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/kladivo_trace_bag_2020-09-22-16-43-31.bag']
#     ]
#
#     bags = [#'/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/gluegun_trace_bag_2020-10-19-11-07-51.bag',
#             #'/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/grout_trace_bag_2020-10-19-15-16-37.bag',
#             #'/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/hammer_trace_bag_2020-10-19-15-22-46.bag',
#             '/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/roller_trace_bag_2020-12-09-15-04-33.bag']
#     # bag_file_path = '/home/behrejan/ros_kinetic_ws/ros_cortex_ws/src/trace_analysis/data/hladitko_trace_bag_2020-09-22-16-02-38.bag.active'
#
#     for bag_file_path in bags:
#         ana = ViveTraceAnalyzer()
#         ana.load_from_bag(bag_file_path)
#         # ana.draw_points()
#         ana.calc_point_density()
#         ana.draw_points_density(density_threshold=5)
#         ana.plot_voxels()
#         ana.show_bb(density_threshold=5)
#         # ana.optimize_bb(density_threshold=18)
#         ana.select_bb(density_threshold=5)
#
#
#     print('ready')
