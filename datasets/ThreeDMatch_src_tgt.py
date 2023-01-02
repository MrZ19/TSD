import os
import os.path
from os.path import join, exists
import numpy as np
import json
import pickle
import random
import open3d as o3d
from utils.pointcloud import make_point_cloud
import torch.utils.data as data
from scipy.spatial.distance import cdist
from geometric_registration.common import get_pcd, get_keypts, get_desc, get_scores, loadlog, build_correspondence


def rotation_matrix(augment_axis, augment_rotation):
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if augment_axis == 1:
        return random.choice([Rx, Ry, Rz]) 
    return Rx @ Ry @ Rz
    
def translation_matrix(augment_translation):
    T = np.random.rand(3) * augment_translation
    return T

class ThreeDMatchTestset(data.Dataset):
    __type__ = 'descriptor'
    def __init__(self, 
                root, 
                downsample=0.03, 
                config=None,
                last_scene=False,
                ):
        self.root = root
        self.downsample = downsample
        self.config = config
        
        # contrainer
        self.points = []
        self.ids_list = []
        self.num_test = 0
        
        self.scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]
        if last_scene == True:
            self.scene_list = self.scene_list[-1:]

        pair_index = []
        pair_index_gt = []
        for scene in self.scene_list:
            self.test_path = f'{self.root}/fragments/{scene}'
            pcd_list = [filename for filename in os.listdir(self.test_path) if filename.endswith('ply')]
            pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
            gtpath = f'geometric_registration/gt_result/{scene}-evaluation/'
            gtLog = loadlog(gtpath)
            num_frag = len(pcd_list)
            for id1 in range(num_frag):
                for id2 in range(id1 + 1, num_frag):
                    cloud_bin_s = f'cloud_bin_{id1}'
                    cloud_bin_t = f'cloud_bin_{id2}'
                    key = f"{id1}_{id2}"

                    if key in gtLog.keys():
                        path_s = os.path.join(self.test_path,cloud_bin_s) #.ply
                        path_t = os.path.join(self.test_path,cloud_bin_t)
                        pair = path_s + '@' + path_t
                        pair_index.append(pair)
                        pair_index_gt.append(gtLog[key])

        self.pair_index = pair_index
        self.pair_index_gt = pair_index_gt
        #     self.num_test += len(pcd_list)

        # for scene in self.scene_list:
        #     self.test_path = f'{self.root}/fragments/{scene}'
        #     pcd_list = [filename for filename in os.listdir(self.test_path) if filename.endswith('ply')]
        #     self.num_test += len(pcd_list)

        #     pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
        #     for i, ind in enumerate(pcd_list):
        #         pcd = o3d.io.read_point_cloud(join(self.test_path, ind))
        #         pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=downsample)
                
        #         # Load points and labels
        #         points = np.array(pcd.points)

        #         self.points += [points]
        #         self.ids_list += [scene + '/' + ind]
        return

    def __getitem__(self, index):
        pair = self.pair_index[index]
        gt_trans = self.pair_index_gt[index]
        src_path, tgt_path = pair.split('@')
        src_pc = o3d.io.read_point_cloud(src_path+'.ply')
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pc, voxel_size=self.downsample)
        src_points = np.array(src_pcd.points).astype(np.float32)
        tgt_pc = o3d.io.read_point_cloud(tgt_path+'.ply')
        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pc, voxel_size=self.downsample)
        tgt_points = np.array(tgt_pcd.points).astype(np.float32)

        feat_src = np.ones_like(src_points[:, :1]).astype(np.float32)
        feat_tgt = np.ones_like(tgt_points[:, :1]).astype(np.float32)

        id1 = int(src_path.split('_')[-1])
        id2 = int(tgt_path.split('_')[-1])
        scene_name = src_path.split('/')[-2]
        if scene_name == '7-scenes-redkitchen':
            scene_id = 0
        if scene_name == 'sun3d-home_at-home_at_scan1_2013_jan_1':
            scene_id = 1
        if scene_name == 'sun3d-home_md-home_md_scan9_2012_sep_30':
            scene_id = 2
        if scene_name == 'sun3d-hotel_uc-scan3':
            scene_id = 3
        if scene_name == 'sun3d-hotel_umd-maryland_hotel1':
            scene_id = 4
        if scene_name == 'sun3d-hotel_umd-maryland_hotel3':
            scene_id = 5
        if scene_name == 'sun3d-mit_76_studyroom-76-1studyroom2':
            scene_id = 6
        if scene_name == 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika':
            scene_id = 7
        return src_points, tgt_points, feat_src, feat_tgt, np.array(gt_trans).astype(np.float32), np.array([scene_id,id1,id2]) # utilize the sel_corr to save gt_trans

    def __len__(self):
        return len(self.pair_index)

if __name__ == "__main__":
    dset = ThreeDMatchDataset(root='/data/3DMatch/', split='train', num_node=64, downsample=0.05, self_augment=True)
    dset[0]
    import pdb
    pdb.set_trace()
