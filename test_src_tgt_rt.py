import os
import open3d as o3d
import argparse
import json
import importlib
import logging
import torch
import numpy as np
from multiprocessing import Process, Manager
from functools import partial
from easydict import EasyDict as edict
from utils.pointcloud import make_point_cloud, make_open3d_point_cloud, make_open3d_feature
from models.architectures_cap import KPFCNN
from utils.timer import Timer, AverageMeter
#from datasets.ThreeDMatch import ThreeDMatchTestset
from datasets.ThreeDMatch_src_tgt import ThreeDMatchTestset
from datasets.dataloader import get_dataloader
from geometric_registration.common import get_pcd, get_keypts, get_desc, get_scores, loadlog, build_correspondence
from tqdm import tqdm

def register_one_scene(inlier_ratio_threshold, distance_threshold, save_path, return_dict, scene):
    gt_matches = 0
    pred_matches = 0
    keyptspath = f"{save_path}/keypoints/{scene}"
    descpath = f"{save_path}/descriptors/{scene}"
    scorepath = f"{save_path}/scores/{scene}"
    gtpath = f'geometric_registration/gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)
    inlier_num_meter, inlier_ratio_meter = AverageMeter(), AverageMeter()
    pcdpath = f"{config.root}/fragments/{scene}/"
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f"{id1}_{id2}"
            if key not in gtLog.keys():
                # skip the pairs that have less than 30% overlap.
                num_inliers = 0
                inlier_ratio = 0
                gt_flag = 0
            else:
                source_keypts = get_keypts(keyptspath, cloud_bin_s)
                target_keypts = get_keypts(keyptspath, cloud_bin_t)
                source_desc = get_desc(descpath, cloud_bin_s, 'D3Feat')
                target_desc = get_desc(descpath, cloud_bin_t, 'D3Feat')
                source_score = get_scores(scorepath, cloud_bin_s, 'D3Feat').squeeze()
                target_score = get_scores(scorepath, cloud_bin_t, 'D3Feat').squeeze()
                source_desc = np.nan_to_num(source_desc)
                target_desc = np.nan_to_num(target_desc)
                
                # randomly select 5000 keypts
                if args.random_points:
                    source_indices = np.random.choice(range(source_keypts.shape[0]), args.num_points)
                    target_indices = np.random.choice(range(target_keypts.shape[0]), args.num_points)
                else:
                    source_indices = np.argsort(source_score)[-args.num_points:]
                    target_indices = np.argsort(target_score)[-args.num_points:]
                source_keypts = source_keypts[source_indices, :]
                source_desc = source_desc[source_indices, :]
                target_keypts = target_keypts[target_indices, :]
                target_desc = target_desc[target_indices, :]
                
                corr = build_correspondence(source_desc, target_desc)

                gt_trans = gtLog[key]
                frag1 = source_keypts[corr[:, 0]]
                frag2_pc = o3d.geometry.PointCloud()
                frag2_pc.points = o3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
                frag2_pc.transform(gt_trans)
                frag2 = np.asarray(frag2_pc.points)
                distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
                num_inliers = np.sum(distance < distance_threshold)
                inlier_ratio = num_inliers / len(distance)
                if inlier_ratio > inlier_ratio_threshold:
                    pred_matches += 1
                gt_matches += 1
                inlier_num_meter.update(num_inliers)
                inlier_ratio_meter.update(inlier_ratio)
    recall = pred_matches * 100.0 / gt_matches
    return_dict[scene] = [recall, inlier_num_meter.avg, inlier_ratio_meter.avg]
    logging.info(f"{scene}: Recall={recall:.2f}%, inlier ratio={inlier_ratio_meter.avg*100:.2f}%, inlier num={inlier_num_meter.avg:.2f}")
    return recall, inlier_num_meter.avg, inlier_ratio_meter.avg


def generate_features(model, dloader, config, chosen_snapshot):
    dataloader_iter = dloader.__iter__()

    descriptor_path = f'{save_path}/descriptors'
    keypoint_path = f'{save_path}/keypoints'
    score_path = f'{save_path}/scores'
    if not os.path.exists(descriptor_path):
        os.mkdir(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.mkdir(keypoint_path)
    if not os.path.exists(score_path):
        os.mkdir(score_path)
    
    # generate descriptors
    recall_list = []
    for scene in dset.scene_list:
        descriptor_path_scene = os.path.join(descriptor_path, scene)
        keypoint_path_scene = os.path.join(keypoint_path, scene)
        score_path_scene = os.path.join(score_path, scene)
        if not os.path.exists(descriptor_path_scene):
            os.mkdir(descriptor_path_scene)
        if not os.path.exists(keypoint_path_scene):
            os.mkdir(keypoint_path_scene)
        if not os.path.exists(score_path_scene):
            os.mkdir(score_path_scene)
        pcdpath = f"{config.root}/fragments/{scene}/"
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        # generate descriptors for each fragment
        for ids in range(num_frag):
            inputs = dataloader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.cuda() for item in v]
                else:
                    inputs[k] = v.cuda()
            features, scores = model(inputs)
            pcd_size = inputs['stack_lengths'][0][0]
            pts = inputs['points'][0][:int(pcd_size)]
            features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]
            # scores = torch.ones_like(features[:, 0:1])
            np.save(f'{descriptor_path_scene}/cloud_bin_{ids}.D3Feat', features.detach().cpu().numpy().astype(np.float32))
            np.save(f'{keypoint_path_scene}/cloud_bin_{ids}', pts.detach().cpu().numpy().astype(np.float32))
            np.save(f'{score_path_scene}/cloud_bin_{ids}', scores.detach().cpu().numpy().astype(np.float32))
            print(f"Generate cloud_bin_{ids} for {scene}")
    

def generate_feature_register_one_scene(inlier_ratio_threshold, distance_threshold, model, dloader, args, chose_snapshot, scene_list):
    num_iter = int(len(dloader.dataset))
    dataloader_iter = dloader.__iter__()

    gt_matches = 0
    pred_matches = 0

    inlier_num_meter, inlier_ratio_meter = AverageMeter(), AverageMeter()
    success_meter, loss_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
    # generate descriptors
    for i in tqdm(range(num_iter)):
        inputs = dataloader_iter.next()
        for k, v in inputs.items(): # load inputs to device.
            if type(v) == list:
                inputs[k] = [item.cuda() for item in v]
            else:
                inputs[k] = v.cuda()
        with torch.no_grad():
            features, scores = model(inputs)
        pcd_size = inputs['stack_lengths'][0][0]
        src_pts = inputs['points'][0][:int(pcd_size)]
        tgt_pts = inputs['points'][0][int(pcd_size):]
        id1 = inputs['dist_keypts'][1]
        id2 = inputs['dist_keypts'][2]
        scene_id = inputs['dist_keypts'][0]
        scene_name = scene_list[scene_id]
        src_features, src_scores = features[:int(pcd_size)], scores[:int(pcd_size)]
        tgt_features, tgt_scores = features[int(pcd_size):], scores[int(pcd_size):]

        # randomly select 5000 keypts
        if args.random_points:
            source_indices = np.random.choice(range(src_pts.shape[0]), args.num_points)
            target_indices = np.random.choice(range(tgt_pts.shape[0]), args.num_points)
        else:
            source_indices = np.argsort(src_scores.squeeze(1).cpu().numpy())[-args.num_points:]
            target_indices = np.argsort(tgt_scores.squeeze(1).cpu().numpy())[-args.num_points:]
        source_keypts = src_pts[source_indices, :].detach().cpu().numpy()
        source_desc = src_features[source_indices, :].detach().cpu().numpy()
        target_keypts = tgt_pts[target_indices, :].detach().cpu().numpy()
        target_desc = tgt_features[target_indices, :].detach().cpu().numpy()
                

        corr = build_correspondence(source_desc, target_desc)

        gt_trans = inputs['corr'].detach().cpu().numpy() #utilize corr position save gt_trans
        if args.data_select == 'kitti':
            T_est = ransac_rt_kitti(source_keypts, target_keypts, source_desc, target_desc)
        elif args.data_select == '3dmatch':
            logpath = f"./geometric_registration/{chose_snapshot}/log_result_pred_5000/{scene_name}-evaluation"
            if not os.path.exists(logpath):
                os.makedirs(logpath)
            T_est = ransac_rt_3dmatch(args, source_keypts, target_keypts, source_desc, target_desc, logpath, id1, id2)
        rte = np.linalg.norm(T_est[:3, 3] - gt_trans[:3, 3])
        rre = np.arccos((np.trace(T_est[:3, :3].transpose() @ gt_trans[:3, :3]) - 1) / 2)

        if rte < 2:
            rte_meter.update(rte)

        if not np.isnan(rre) and rre < np.pi / 180 * 5:
            rre_meter.update(rre * 180 / np.pi)

        if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
            success_meter.update(1)
        else:
            success_meter.update(0)
            logging.info(f"Failed with RTE: {rte}, RRE: {rre * 180 / np.pi}")

        frag1 = source_keypts[corr[:, 0]]
        frag2_pc = o3d.geometry.PointCloud()
        frag2_pc.points = o3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
        frag2_pc.transform(gt_trans) ##
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        num_inliers = np.sum(distance < distance_threshold)
        inlier_ratio = num_inliers / len(distance)
        if inlier_ratio > inlier_ratio_threshold:
            pred_matches += 1
        gt_matches += 1
        inlier_num_meter.update(num_inliers)
        inlier_ratio_meter.update(inlier_ratio)

    recall = pred_matches * 100.0 / gt_matches
    # return_dict[scene] = [recall, inlier_num_meter.avg, inlier_ratio_meter.avg]
    logging.info(f"Recall={recall:.2f}%, inlier ratio={inlier_ratio_meter.avg*100:.2f}%, inlier num={inlier_num_meter.avg:.2f}")
    return recall, inlier_num_meter.avg, inlier_ratio_meter.avg


def ransac_rt_kitti(anc_points, pos_points, anc_features, pos_features):
    pcd0 = make_open3d_point_cloud(anc_points)
    pcd1 = make_open3d_point_cloud(pos_points)
    feat0 = make_open3d_feature(anc_features, 32, anc_features.shape[0])
    feat1 = make_open3d_feature(pos_features, 32, pos_features.shape[0])

    distance_threshold = 0.05
    ransac_result = o3d.registration.registration_ransac_based_on_feature_matching(
                    pcd0, pcd1, feat0, feat1, distance_threshold,
                    o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                        o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                        o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                    o3d.registration.RANSACConvergenceCriteria(50000, 1000))
                # print(ransac_result)
    T_ransac = ransac_result.transformation.astype(np.float32)
    return T_ransac

def ransac_rt_3dmatch(config, source_keypts, target_keypts, source_desc, target_desc, logpath, id1, id2):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_keypts)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_keypts)
    s_desc = o3d.pipelines.registration.Feature()
    s_desc.data = source_desc.T
    t_desc = o3d.pipelines.registration.Feature()
    t_desc.data = target_desc.T
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, s_desc, t_desc, True,
            0.05,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

    # write the transformation matrix into .log file for evaluation.
    with open(os.path.join(logpath, f'{config.chosen_snapshot}.log'), 'a+') as f:
        trans = result.transformation
        trans = np.linalg.inv(trans)
        s1 = f'{id1}\t {id2}\t  37\n'
        f.write(s1)
        f.write(f"{trans[0,0]}\t {trans[0,1]}\t {trans[0,2]}\t {trans[0,3]}\t \n")
        f.write(f"{trans[1,0]}\t {trans[1,1]}\t {trans[1,2]}\t {trans[1,3]}\t \n")
        f.write(f"{trans[2,0]}\t {trans[2,1]}\t {trans[2,2]}\t {trans[2,3]}\t \n")
        f.write(f"{trans[3,0]}\t {trans[3,1]}\t {trans[3,2]}\t {trans[3,3]}\t \n")
    return trans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='D3Feat_multi_srctgt_v0_resume10122042', type=str, help='snapshot dir')
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.10, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=5000, type=int)
    parser.add_argument('--generate_features', default=True, action='store_true')
    parser.add_argument('--data_select',default='3dmatch', type=str)
    args = parser.parse_args()
    if args.random_points:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-rand-{args.num_points}.log'
    else:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-pred-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO, 
        filename=log_filename, 
        filemode='w', 
        format="")


    config_path = f'./data/D3Feat/snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    # create model 
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')
    
    model = KPFCNN(config)
    model.load_state_dict(torch.load(f'./data/D3Feat/snapshot/{args.chosen_snapshot}/models/model_best_acc.pth')['state_dict'])
    print(f"Load weight from snapshot/{args.chosen_snapshot}/models/model_best_acc.pth")
    #with torch.no_grad():
    model.eval()


    save_path = f'geometric_registration/{args.chosen_snapshot}'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)


    #if args.generate_features:
    if True:
        dset = ThreeDMatchTestset(root=config.root,
                            downsample=config.downsample,
                            config=config,
                            last_scene=False,
                        )
        dloader, _ = get_dataloader(dataset=dset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    )
        # with torch.no_grad():
        #     generate_features(model.cuda(), dloader, config, args.chosen_snapshot)
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    result = generate_feature_register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, model.cuda(), dloader, args, args.chosen_snapshot, scene_list)
    # register each pair of fragments in scenes using multiprocessing.
    # scene_list = [
    #     '7-scenes-redkitchen',
    #     'sun3d-home_at-home_at_scan1_2013_jan_1',
    #     'sun3d-home_md-home_md_scan9_2012_sep_30',
    #     'sun3d-hotel_uc-scan3',
    #     'sun3d-hotel_umd-maryland_hotel1',
    #     'sun3d-hotel_umd-maryland_hotel3',
    #     'sun3d-mit_76_studyroom-76-1studyroom2',
    #     'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    # ]
    # return_dict = Manager().dict()
    # # register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene_list[0])
    # jobs = []
    # for scene in scene_list:
    #     p = Process(target=register_one_scene, args=(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene))
    #     jobs.append(p)
    #     p.start()
    
    # for proc in jobs:
    #     proc.join()
    # result = []
    # for scene in scene_list:
    #     p = register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene)
    #     result.append(p)

    # recalls = [v[0] for v in result]
    # inlier_nums = [v[1] for v in result]
    # inlier_ratios = [v[2] for v in result]

    # # recalls = [v[0] for k, v in return_dict.items()]
    # # inlier_nums = [v[1] for k, v in return_dict.items()]
    # # inlier_ratios = [v[2] for k, v in return_dict.items()]

    logging.info("*" * 40)
    # logging.info(recalls)
    logging.info(f"All 8 scene, average recall: {result[0]:.2f}%")
    logging.info(f"All 8 scene, average num inliers: {result[1]:.2f}")
    logging.info(f"All 8 scene, average num inliers ratio: {result[2]*100:.2f}%")
