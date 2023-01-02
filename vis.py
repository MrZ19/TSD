#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import open3d as o3d
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import manifold
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def genereate_correspondence_line_set(src, target):
    points = np.concatenate((src, target), axis=0)
    lines = [[i,i+src.shape[0]] for i in range(src.shape[0])]
    colors = [[0.47, 0.53, 0.7] for i in range(len(lines))] # 0.69, 0.76, 0.87 / 0.9, 0, 0
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def verify_SVD(src, transformed_src, virtual_points, rotation_ab_pred, translation_ab_pred):

    src_centered = (src - src.mean(axis=0)).T
    transformed_src_centered = (transformed_src - transformed_src.mean(axis=0)).T
    virtual_points_centered  = (virtual_points  - virtual_points.mean(axis=0)).T

    H1 = np.dot(src_centered, transformed_src_centered.T)
    u1, s1, v1t = np.linalg.svd(H1, full_matrices=True)
    print("\nH in SVD -- src & transformed_src:\n", H1)
    print("\ns in SVD -- src & transformed_src:\n", s1)
    print("\nu in SVD -- src & transformed_src:\n", u1)
    print("\nvt in SVD -- src & transformed_src:\n", v1t)
    R1 = np.dot(v1t.T, u1.T)
    if np.linalg.det(R1) < 0:
        v1t[2,:] *= -1
        R1 = np.dot(v1t.T, u1.T)
    t1 = transformed_src.mean(axis=0) - np.dot(R1, src.mean(axis=0))

    H2 = np.dot(src_centered, virtual_points_centered.T)
    u2, s2, v2t = np.linalg.svd(H2, full_matrices=True)
    print("\nH in SVD -- src & virtual_points:\n", H2)
    print("\ns in SVD -- src & virtual_points:\n", s2)
    print("\nu in SVD -- src & virtual_points:\n", u2)
    print("\nvt in SVD -- src & virtual_points:\n", v2t)
    R2 = np.dot(v2t.T, u2.T)
    if np.linalg.det(R2) < 0:
        v2t[2,:] *= -1
        R2 = np.dot(v2t.T, u2.T)
    t2 =  virtual_points.mean(axis=0) - np.dot(R2, src.mean(axis=0))

    print('\nrotation_ab_pred:\n', rotation_ab_pred)
    print('translation_ab_pred:\n', translation_ab_pred)
    print('\nR1 - src & transformed_src:\n', R1)
    print('t1 - src & transformed_src:\n', t1)
    print('\nR2 - src & virtual_points:\n', R2)
    print('t2 - src & virtual_points:\n', t2)

    H_recover = u2 @ np.diag(s1) @ v2t
    print("\nH_recover:\n",H_recover)
    point_recover = np.linalg.pinv(src_centered) @ H_recover #+ transformed_src.mean(axis=0)

    return R1, t1, R2, t2, point_recover

def visualize_trans_result(src, transformed_src, target, rotation_ab_pred, translation_ab_pred, rotation_gt, translation_gt):
    # afer permute -- (batch_size, point_nunber, 3xyz)
    src = src.permute(0,2,1).cpu().numpy()
    transformed_src = transformed_src.permute(0,2,1).cpu().detach().numpy()
    target = target.permute(0,2,1).cpu().numpy()
    #virtual_corr_points = virtual_corr_points.permute(0,2,1).cpu().detach().numpy()
    #scores_ab_pred = scores_ab_pred.cpu().detach().numpy()
    rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()
    translation_ab_pred = translation_ab_pred.detach().cpu().numpy()

    rotation_gt = rotation_gt.detach().cpu().numpy()
    translation_gt = translation_gt.detach().cpu().numpy()

    for _src, _transformed_src, _target, _rotation_ab_pred, _translation_ab_pred, _rotation_gt, _translation_gt in \
        zip(src, transformed_src, target, rotation_ab_pred, translation_ab_pred, rotation_gt, translation_gt):

        # 已验证 _transformed_src 是 _src 经过 _rotation_ab_pred, _translation_ab_pred 变换得到的
        # R1, t1, R2, t2, point_recover = \
        #     verify_SVD(_src, _transformed_src, _virtual_corr, _rotation_ab_pred, _translation_ab_pred)

        src_pcd = o3d.geometry.PointCloud()
        _src[:,0] -= 2.0
        src_pcd.points = o3d.utility.Vector3dVector(_src)
        src_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 255, 0]]), _src.shape[0], axis=0))

        # _transformed_src = (_transformed_src - _transformed_src.mean(axis=0))
        transformed_src_pcd = o3d.geometry.PointCloud()
        transformed_src_pcd.points = o3d.utility.Vector3dVector(_transformed_src)
        transformed_src_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[255, 0, 0]]), _transformed_src.shape[0], axis=0))

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(_target)
        target_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 0, 255]]), _target.shape[0], axis=0))

        # virtual_corr_pcd = o3d.geometry.PointCloud()
        # virtual_corr_pcd.points = o3d.utility.Vector3dVector(_virtual_corr)
        # virtual_corr_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[255, 0, 255]]), _virtual_corr.shape[0], axis=0))
        
        # point_recover_pcd = o3d.geometry.PointCloud()
        # point_recover_pcd.points = o3d.utility.Vector3dVector(point_recover)
        # point_recover_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 0, 0]]), point_recover.shape[0], axis=0))

        #correspondence_line_set = genereate_correspondence_line_set(_src, _virtual_corr)#point_recover)
        
        # visualize_scores_distribution(_scores_ab_pred)
        #o3d.visualization.draw_geometries([src_pcd, transformed_src_pcd, target_pcd, virtual_corr_pcd, correspondence_line_set]) #, point_recover_pcd])
        calculate_error(_rotation_gt,_translation_gt,_rotation_ab_pred,_translation_ab_pred)
        o3d.visualization.draw_geometries([src_pcd,transformed_src_pcd, target_pcd])
        pass

def visualize_scores_distribution(_scores_ab_pred):
    # pass
    
    _scores_ab_pred.sort(axis=1)
    x_axis = list(range(_scores_ab_pred.shape[1]))
    for i in range(_scores_ab_pred.shape[0]):
        plt.plot(x_axis, _scores_ab_pred[i])
    
    plt.xlabel('xi in source point')
    plt.ylabel('Probability of point in target cloud')
    plt.show()

def calculate_error(R1,t1,R2,t2):
    diff_r = np.matmul(R1, R2.T)
    angle = np.arccos((np.trace(diff_r)-1)/2)*180.0/np.pi
    t_error = np.sqrt(np.sum((t1 - t2)**2))
    print("angle:{}".format(angle))
    print("t:{}".format(t_error))

def visualize_pc(src, src_I, target, target_I):
    
    src = src.permute(0,2,1).cpu().numpy()
    src_I = src_I.permute(0,2,1).cpu().detach().numpy()
    target = target.permute(0,2,1).cpu().numpy()
    target_I = target_I.permute(0,2,1).cpu().detach().numpy()
    print(src.shape)


    for _src, _src_I, _target, _target_I in \
        zip(src, src_I, target, target_I):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        
        src_pcd = o3d.geometry.PointCloud()
        #_src[:,0] -= 2.0
        src_pcd.points = o3d.utility.Vector3dVector(_src)
        src_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 255, 0]]), _src.shape[0], axis=0))

        
        src_I_pcd = o3d.geometry.PointCloud()
        #_src_I[:,0] -= 4.0
        src_I_pcd.points = o3d.utility.Vector3dVector(_src_I)
        src_I_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[255, 0, 0]]), _src_I.shape[0], axis=0))

        target_pcd = o3d.geometry.PointCloud()
        _target[:,0] -= 6.0
        target_pcd.points = o3d.utility.Vector3dVector(_target)
        target_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 0, 255]]), _target.shape[0], axis=0))

        target_I_pcd = o3d.geometry.PointCloud()
        target_I_pcd.points = o3d.utility.Vector3dVector(_target_I)
        target_I_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 255, 255]]), _target_I.shape[0], axis=0))

        #correspondence_line_set = genereate_correspondence_line_set(_src, _virtual_corr)#point_recover)
        
        # visualize_scores_distribution(_scores_ab_pred)
        #o3d.visualization.draw_geometries([src_pcd, transformed_src_pcd, target_pcd, virtual_corr_pcd, correspondence_line_set]) #, point_recover_pcd])
        #calculate_error(_rotation_gt,_translation_gt,_rotation_ab_pred,_translation_ab_pred) , target_I_pcd
        
        #o3d.visualization.draw_geometries([src_pcd, src_I_pcd, target_pcd, target_I_pcd])
        o3d.visualization.draw_geometries([coordinate_frame, src_pcd, src_I_pcd, target_pcd, target_I_pcd])
        o3d.visualization.draw_geometries([coordinate_frame, src_pcd, src_I_pcd])
        pass


def vis_feature_distribution(src_points, src_des, tgt_points, tgt_des, gt_trans):
    # src_points: B,N,3 coordinates
    # src_des: B,N,F    feature descriptor
    # tgt_points: B,N,3 coordinates
    # tgt_des: B,N,F    feature descriptor
    _src = src_points#.permute(0,2,1).cpu().numpy()
    _srcD = src_des#.permute(0,2,1).cpu().detach().numpy()
    _tgt = tgt_points#.permute(0,2,1).cpu().numpy()
    _tgtD = tgt_des#.permute(0,2,1).cpu().detach().numpy()

    srcD_color = TSNE_feature(_srcD)
    tgtD_color = TSNE_feature(_tgtD)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    src_pcd = o3d.geometry.PointCloud()
    _src[:,0] -= 4.0
    src_pcd.points = o3d.utility.Vector3dVector(_src)
    #src_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[0, 255, 0]]), _src.shape[0], axis=0))
    src_pcd.colors = o3d.utility.Vector3dVector(srcD_color)

        
    tgt_pcd = o3d.geometry.PointCloud()
    #_tgt[:,0] -= 4.0
    tgt_pcd.points = o3d.utility.Vector3dVector(_tgt)
    tgt_pcd.transform(gt_trans)
    #tgt_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.asarray([[255, 0, 0]]), _src_I.shape[0], axis=0))
    tgt_pcd.colors = o3d.utility.Vector3dVector(tgtD_color)

    o3d.visualization.draw_geometries([coordinate_frame, src_pcd, tgt_pcd])
    pass 

def TSNE_feature(Des):
    tsne = manifold.TSNE(n_components=3,init='pca', random_state=501)
    X_tsne = tsne.fit_transform(Des)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne-x_min)/(x_max-x_min)
    return X_norm #* 255.0