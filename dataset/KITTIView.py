"""
Written by Heng Fan
Update by x.l
The KITTI Visual View powered by MayAvi
"""
import os
import cv2 as cv
import mayavi.mlab as mlab
from KITTI import KITTI
from utility import *
from easydict import EasyDict as edict

class KittiView:
    def __init__(self, dataPath):
        self.data = KITTI(dataPath)
    
    def _map(self,vid_id):
        sequence = self.data.sequence_list[vid_id]
        kip = edict({
            'sequence_name': sequence['name'],       # eg. '0001'
            'img_list': sequence['img_list'],        # get image list of this sequence
            'labels': sequence['label_list'],        # get label list of this sequence
            'calib': sequence['calib'],              # get the calibration matrices of this sequence
            'colors': self.data.colors,
            'categories': self.data.categories,
            'pcloud_list':sequence.pcloud_list
        })                                  
        return kip

    def show_sequence_rgb(self, vid_id, vis_2dbox=False, vis_3dbox=False, save_img=False, save_path=None, wait_time=30):
        """
        visualize the sequence in RGB
        :param vid_id: id of the sequence, starting from 0
        :return: none
        """
        kip = self._map(vid_id) # get mapped kitti

        assert vid_id>=0 and vid_id<len(self.data.sequence_list), \
            'The id of the sequence should be in the range [0, {}]'.format(str(self.sequence_num-1))
        assert len(kip.img_list) == len(kip.labels), 'The number of image and number of labels do NOT match!'
        assert not(vis_2dbox == True and vis_3dbox == True), 'It is NOT good to visualize both 2D and 3D boxes simultaneously!'

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_2dbox:
                    save_path = os.path.join('./seq_camera_vis', kip.sequence_name+'_2D_box')
                elif vis_3dbox:
                    save_path = os.path.join('./seq_camera_vis', kip.sequence_name+'_3D_box')
                else:
                    save_path = os.path.join('./seq_camera_vis', kip.sequence_name+'_no_box')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # show the sequence
        for img_name, img_label in zip(kip.img_list, kip.labels):
            img = cv.imread(img_name)   # BGR image format
            thickness = 2

            # visualize 2d boxes in the image
            if vis_2dbox:
                # load and show object bboxes
                for object in img_label:
                    object_type = object['object_type']
                    bbox = object['bbox']
                    bbox = [int(tmp) for tmp in bbox]
                    bbox_color = kip.colors[kip.categories.index(object_type)]
                    bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])
                    cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=bbox_color, thickness=thickness)

                    cv.putText(img, text=object_type + '-ID: ' + str(object['id']), org=(bbox[0], bbox[1] - 5),
                                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)

            # visualize 3d boxes in the image
            if vis_3dbox:
                # load and show object bboxes
                for object in img_label:
                    object_type = object['object_type']
                    bbox_color = kip.colors[kip.categories.index(object_type)]
                    bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])
                    #image size: 1392*512
                    corners_3d_img = transform_3dbox_to_image(object['dimension'], object['location'], object['rotation'], kip.calib)

                    if corners_3d_img is None:
                        # None means object is behind the camera, and ignore this object.
                        continue
                    else:
                        corners_3d_img = corners_3d_img.astype(int)

                        # draw lines in the image
                        # p0-p1, p1-p2, p2-p3, p3-p0  底面
                        for (x1,x2) in zip(range(4),[1,2,3,0]):
                            cv.line(img, (corners_3d_img[x1, 0], corners_3d_img[x1, 1]),
                                (corners_3d_img[x2, 0], corners_3d_img[x2, 1]), color=bbox_color, thickness=thickness)
                        # p4-p5, p5-p6, p6-p7, p7-p0   顶面
                        for (x1,x2) in zip(range(4,8),[5,6,7,4]):
                            cv.line(img, (corners_3d_img[x1, 0], corners_3d_img[x1, 1]),
                                (corners_3d_img[x2, 0], corners_3d_img[x2, 1]), color=bbox_color, thickness=thickness)
                        # p0-p4, p1-p5, p2-p6, p3-p7    四条棱
                        for (x1,x2) in zip(range(4),[4,5,6,7]):
                            cv.line(img, (corners_3d_img[x1, 0], corners_3d_img[x1, 1]),
                                (corners_3d_img[x2, 0], corners_3d_img[x2, 1]), color=bbox_color, thickness=thickness)
                        # draw front lines 画X
                        for (x1,x2) in zip(range(2),[5,4]):
                            cv.line(img, (corners_3d_img[x1, 0], corners_3d_img[x1, 1]),
                                (corners_3d_img[x2, 0], corners_3d_img[x2, 1]), color=bbox_color, thickness=thickness)
             
                        cv.putText(img, text=object_type + '-ID: ' + str(object['id']), org=(corners_3d_img[4, 0], corners_3d_img[4, 1]-5),
                                    fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)

            cv.imshow('Play {}'.format(kip.sequence_name), img)
            # save visualization image if you want
            if save_img:
                cv.imwrite(os.path.join(save_path, img_name.split('/')[-1].split('.')[0] + '.png'), img)
            cv.waitKey(wait_time)
            cv.destroyAllWindows()

    def show_sequence_pointcloud(self, vid_id, img_region=False, vis_box=False, save_img=False, save_path=None):
        """
        visualize the sequence in point cloud
        :param vid_id: id of the sequence, starting from 0
        :param img_region: only show point clouds in RGB image
        :param vis_box: show 3D boxes or not
        :return: none
        """
        kip = self._map(vid_id) # get mapped kitti
        assert 0 <= vid_id < len(self.data.sequence_list), 'The sequence id should be in [0, {}]'.format(str(self.sequence_num - 1))

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_box:
                    save_path = os.path.join('./seq_pointcloud_vis', kip.sequence_name+'_3D_box')
                else:
                    save_path = os.path.join('./seq_pointcloud_vis', kip.sequence_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
        pcloud_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
        # plt = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], mode='point', figure=pcloud_fig) #画点

        for pcloud_name in kip.pcloud_list:
            # clear
            mlab.clf()

            # BE CAREFUL!
            # the reason why doing so is because there are bin files missing in some sequences (e.g., sequence 0001)
            # e.g., in label file, the seuqnece is: 000001, 000002, 000003, 000004, 000005
            # but in bin file, the sequence is:     000001, 000004, 000005
            img_label = kip.labels[int(pcloud_name.split('/')[-1].split('.')[0])]

            # load point cloud
            # point[:, 0]: x; point[:, 1]: y; point[:, 2]: z; point[:, 3]: reflectance information
            pcloud = np.fromfile(pcloud_name, dtype=np.float32).reshape(-1, 4)

            # remove point clouds not in RBG image
            if img_region:
                # velodyne coordinate to camera 0 coordinate
                pcloud_C2_depth, pcloud_C2 = velodyne_to_camera_2(pcloud, kip.calib)

                # remove points out of image
                pcloud_in_img = remove_cloudpoints_out_of_image(pcloud_C2_depth, pcloud_C2, pcloud, kip.img_size)
                pcloud = pcloud_in_img

            # show point cloud
            plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], np.arange(len(pcloud)), mode='point', figure=pcloud_fig)
            # plot = mlab.points3d(pcloud[:,np 0], pcloud[:, 1], pcloud[:, 2], mode='point', figure=pcloud_fig)

            # load and show 3d boxes
            if vis_box:
                for object in img_label:
                    object_type = object['object_type']
                    bbox_color = kip.colors[kip.categories.index(object_type)]
                    bbox_color = (bbox_color[2]/255, bbox_color[1]/255, bbox_color[0]/255)
                    corners_3d = transform_3dbox_to_pointcloud(object['dimension'], object['location'], object['rotation'])
                    center_3d = transform_3dbox_to_pointcloud(object['dimension'], object['location'], object['rotation'], True)
                    # draw lines
                    # a utility function to draw a line
                    def draw_line_3d(p1, p2, line_color=(0, 0, 0), fig=None):
                        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=line_color, tube_radius=None, line_width=3, figure=fig)

                    # draw the bootom lines, up lines  底面和顶面
                    for (x1,x2) in zip(range(8),[1,2,3,0,5,6,7,4]):
                        draw_line_3d(corners_3d[x1], corners_3d[x2], bbox_color)

                    # # draw the vertical lines 4条棱
                    for (x1,x2) in zip(range(4,8),range(4)):
                        draw_line_3d(corners_3d[x1], corners_3d[x2], bbox_color)

                    # draw front lines 画X
                    # draw_line_3d(corners_3d[4], corners_3d[1], bbox_color)
                    # draw_line_3d(corners_3d[5], corners_3d[0], bbox_color)

                    # draw center 画中心点
                    x = center_3d[:, 0]  # x position of point
                    y = center_3d[:, 1]  # y position of point
                    z = center_3d[:, 2]  # z position of point
                    for x in range(8):
                        draw_line_3d(corners_3d[x], center_3d[0], bbox_color)

                    # mlab.text3d(x=corners_3d[5, 0], y=corners_3d[5, 1], z=corners_3d[5, 2], \
                    #             text=object_type+'-ID: '+str(object['id']), color=bbox_color, scale=0.35)

            # fix the view of the camera
            mlab.view(azimuth=180, distance=30, elevation=60, focalpoint=np.mean(pcloud, axis=0)[:-1])
            # mlab.show()

            if save_img:
                mlab.savefig(filename=os.path.join(save_path, pcloud_name.split('/')[-1].split('.')[0] + '.png'))
            else:
                mlab.savefig(filename='temp_img.png')  # save the visualization image (this line is necessary for visualization)

        # mlab.show()   # do NOT use this line, as it will get the focus and pause the code
        mlab.close(all=True)
        if not save_img:
            os.remove(path='temp_img.png')  # remove temp image file

    def show_sequence_BEV(self, vid_id, img_region=False, vis_box=False, save_img=False, save_path=None):
        """
        visualize the sequence in bird's eye view
        :param vid_id: id of the sequence, starting from 0
        :param img_region: only show point clouds in RGB image
        :param vis_3dbox: show 3D boxes or not
        :return: none
        """
        kip = self._map(vid_id) # get mapped kitti
        assert 0 <= vid_id < len(self.data.sequence_list), 'The sequence id should be in [0, {}]'.format(str(self.data.sequence_num - 1))

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_box:
                    save_path = os.path.join('./seq_BEV_vis', kip.sequence_name + '_BEV_box')
                else:
                    save_path = os.path.join('./seq_BEV_vis', kip.sequence_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # visualization
        pcloud_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
        for pcloud_name in kip.pcloud_list:
            # clear
            mlab.clf()

            # BE CAREFUL!
            # the reason why doing so is because there are bin files missing in some sequences (e.g., sequence 0001)
            # e.g., in label file, the seuqnece is: 000001, 000002, 000003, 000004, 000005
            # but in bin file, the sequence is:     000001, 000004, 000005
            img_label = kip.labels[int(pcloud_name.split('/')[-1].split('.')[0])]

            # load point cloud
            # point[:, 0]: x; point[:, 1]: y; point[:, 2]: z; point[:, 3]: reflectance information
            pcloud = np.fromfile(pcloud_name, dtype=np.float32).reshape(-1, 4)

            # remove point clouds not in RBG image
            if img_region:
                # velodyne coordinate to camera 0 coordinate
                pcloud_C2_depth, pcloud_C2 = velodyne_to_camera_2(pcloud, kip.calib)

                # remove points out of image
                pcloud_in_img = remove_cloudpoints_out_of_image(pcloud_C2_depth, pcloud_C2, pcloud, kip.img_size)
                pcloud = pcloud_in_img

            # show point cloud
            plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], np.arange(len(pcloud)), mode='point', figure=pcloud_fig)

            # load and show 3d boxes
            if vis_box:
                for object in img_label:
                    object_type = object['object_type']
                    bbox_color = kip.colors[kip.categories.index(object_type)]
                    bbox_color = (bbox_color[2]/255, bbox_color[1]/255, bbox_color[0]/255)
                    corners_3d = transform_3dbox_to_pointcloud(object['dimension'], object['location'], object['rotation'])

                    # draw lines
                    # a utility function to draw a line
                    def draw_line_3d(p1, p2, line_color=(0, 0, 0), fig=None):
                        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=line_color, tube_radius=None, line_width=4, figure=fig)

                    # draw the lines in X-Y space
                    for (x1,x2) in zip(range(4,8),[5,6,7,4]):
                        draw_line_3d(corners_3d[x1], corners_3d[x2], bbox_color)

                    mlab.text3d(x=corners_3d[7, 0], y=corners_3d[7, 1]-0.5, z=corners_3d[7, 2], \
                                text=object_type + '-ID: ' + str(object['id']), color=bbox_color, scale=0.7)

            # fix the view of the camera
            mlab.view(azimuth=180, distance=100, elevation=0, focalpoint=np.mean(pcloud, axis=0)[:-1])
            if save_img:
                mlab.savefig(filename=os.path.join(save_path, pcloud_name.split('/')[-1].split('.')[0] + '.png'))
            else:
                mlab.savefig(filename='temp_img.png')  # save the visualization image (this line is necessary for visualization)

        # mlab.show()   # do NOT use this line, as it will get the focus and pause the code
        mlab.close(all=True)
        if not save_img:
            os.remove(path='temp_img.png')  # remove temp image file


# # This is for debug
if __name__ == '__main__':
    kitti_path = '/Users/avivaliu/Visualize-KITTI-Objects-in-Videos/data/KITTI'
    kittiView = KittiView(kitti_path)
    # kittiView.show_sequence_rgb(0, vis_2dbox=False, vis_3dbox=True, save_img=True)
    kittiView.show_sequence_pointcloud(0, img_region=False, vis_box=True)
    # kittiView.show_sequence_BEV(0, vis_box=True, save_img=True)
    print('end!')
    