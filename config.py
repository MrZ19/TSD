import argparse
import time
import os

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')

experiment_id = "D3Feat_multi_srctgt_v0_resume" + time.strftime('%m%d%H%M')
# snapshot configurations
snapshot_arg = add_argument_group('Snapshot')
snapshot_arg.add_argument('--snapshot_dir', type=str, default=f'./data/D3Feat/snapshot/{experiment_id}')
snapshot_arg.add_argument('--tboard_dir', type=str, default=f'./data/D3Feat/tensorboard/{experiment_id}')
snapshot_arg.add_argument('--snapshot_interval', type=int, default=100)
snapshot_arg.add_argument('--save_dir', type=str, default=os.path.join(f'./data/D3Feat/snapshot/{experiment_id}', 'models/'))

# Network configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--num_layers', type=int, default=5)
net_arg.add_argument('--in_points_dim', type=int, default=3)
net_arg.add_argument('--first_features_dim', type=int, default=128)
net_arg.add_argument('--first_subsampling_dl', type=float, default=0.03)
net_arg.add_argument('--in_features_dim', type=int, default=1)
net_arg.add_argument('--conv_radius', type=float, default=2.5)
net_arg.add_argument('--deform_radius', type=float, default=5.0)
# net_arg.add_argument('--density_parameter', type=float, default=5.0) # previous name for deform_radius
net_arg.add_argument('--num_kernel_points', type=int, default=15)
net_arg.add_argument('--KP_extent', type=float, default=2.0)
net_arg.add_argument('--KP_influence', type=str, default='linear')
# net_arg.add_argument('--convolution_mode', type=str, default='sum', choices=['closest', 'sum'])
net_arg.add_argument('--aggregation_mode', type=str, default='sum', choices=['closest', 'sum'])
net_arg.add_argument('--fixed_kernel_points', type=str, default='center', choices=['center', 'verticals', 'none'])
net_arg.add_argument('--use_batch_norm', type=str2bool, default=False)
net_arg.add_argument('--batch_norm_momentum', type=float, default=0.02)
net_arg.add_argument('--deformable', type=str2bool, default=False)
net_arg.add_argument('--modulated', type=str2bool, default=False)
 
# Loss configurations
loss_arg = add_argument_group('Loss')
loss_arg.add_argument('--dist_type', type=str, default='euclidean')
loss_arg.add_argument('--desc_loss', type=str, default='circle', choices=['contrastive', 'circle'])
loss_arg.add_argument('--pos_margin', type=float, default=0.1)
loss_arg.add_argument('--neg_margin', type=float, default=1.4)
loss_arg.add_argument('--m', type=float, default=0.1)
loss_arg.add_argument('--log_scale', type=float, default=10)
loss_arg.add_argument('--safe_radius', type=float, default=0.1)
loss_arg.add_argument('--det_loss', type=str, default='score')
loss_arg.add_argument('--desc_loss_weight', type=float, default=1.0)
loss_arg.add_argument('--det_loss_weight', type=float, default=1.0)

# Optimizer configurations
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'ADAM'])
opt_arg.add_argument('--max_epoch', type=int, default=150)
opt_arg.add_argument('--training_max_iter', type=int, default=3500)
opt_arg.add_argument('--val_max_iter', type=int, default=500)
opt_arg.add_argument('--lr', type=float, default=0.01)
opt_arg.add_argument('--weight_decay', type=float, default=1e-6)
opt_arg.add_argument('--momentum', type=float, default=0.98)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--scheduler_gamma', type=float, default=0.1**(1/80))
opt_arg.add_argument('--scheduler_interval', type=int, default=1)
opt_arg.add_argument('--grad_clip_norm', type=float, default=100.0)

# Dataset and dataloader configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--root', type=str, default='../D3feat_3dmatch_trainval_3cm/')
data_arg.add_argument('--num_node', type=int, default=128)
data_arg.add_argument('--downsample', type=float, default=0.03)
data_arg.add_argument('--self_augment', type=str2bool, default=False)
data_arg.add_argument('--augment_noise', type=float, default=0.005)
data_arg.add_argument('--augment_axis', type=int, default=1)
data_arg.add_argument('--augment_rotation', type=float, default=1.0, help='rotation angle = num * 2pi') 
data_arg.add_argument('--augment_translation', type=float, default=0.5, help='translation = num (m)')
data_arg.add_argument('--batch_size', type=int, default=1)
data_arg.add_argument('--num_workers', type=int, default=16)

# Other configurations
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--gpu_mode', type=str2bool, default=True)
misc_arg.add_argument('--verbose', type=str2bool, default=True)
misc_arg.add_argument('--pretrain', type=str, default='/home1/zhangzhiyuan/research/OurD3feat/D3Feat_fusion/data/D3Feat/snapshot/D3Feat_multi_srctgt_v010101437/models/model_best_acc.pth')

# parameters
dcp_arg = add_argument_group('DCP')
dcp_arg.add_argument('--emb_nn', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
dcp_arg.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
dcp_arg.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
dcp_arg.add_argument('--emb_dims', default=[2048, 1024, 512, 256, 32],
                        help='Dimension of embeddings')
dcp_arg.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
dcp_arg.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
dcp_arg.add_argument('--ff_dims', default=[4096, 2048, 1024, 512, 64],
                        help='Num of dimensions of fc in transformer')
dcp_arg.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')

# capsule network for key probability of each points
cap_arg = add_argument_group('Cap')
cap_arg.add_argument('--latent_caps_size', type=int, default=50)
cap_arg.add_argument('--latent_vec_size', type=int, default=32)
cap_arg.add_argument('--prim_caps_size', type=int, default=50)
cap_arg.add_argument('--prim_vec_size', type=int, default=32)

# cross aggration
cross_arg = add_argument_group('Cross')
cross_arg.add_argument('--assignment', type=str, default='dynamic_routing')

def get_config():
  args = parser.parse_args()
  return args
