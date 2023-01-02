import open3d as o3d 
import torch 
import numpy as np 
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.blocks import closest_pool, gather
from models.pointnet2_utils import PointNetFeaturePropagation
from tqdm import tqdm
### class: assignment ### 
class Assign(nn.Module):
    def __init__(self):
        super(Assign, self).__init__()
        self.map_2048_32 = MLP2048_32()
        self.map_1024_32 = MLP1024_32()
        self.map_512_32 = MLP512_32()
        self.map_256_32 = MLP256_32()

        self.map_2048_16 = MLP2048_16()
        self.map_1024_16 = MLP1024_16()
        self.map_512_16 = MLP512_16()
        self.map_256_16 = MLP256_16()
        self.FP_upsample = PointNetFeaturePropagation(in_channel=32, mlp=[32, 32])

        latent_vec_size, prim_caps_size, prim_vec_size = 32, 5, 32, 

        self.latent_caps_learner = LatentCapsLayer(1, prim_caps_size, prim_vec_size, latent_vec_size)

    def forward(self, assign_choice, batch, x_decoder, st):
        x = x_decoder[-1]
        x_final = x.clone()
        x_decoder = x_decoder[:4]
        if assign_choice == 'attention':
            x_de = x_decoder[0]
            x_de = self.map_2048_32(x_de.unsqueeze(0).transpose(2,1))
            x_update = scaled_attention(x_de.squeeze(0).transpose(1,0), x)
            x_final = x_final + x_update

            x_de = x_decoder[1]
            x_de = self.map_1024_32(x_de.unsqueeze(0).transpose(2,1))
            x_update = scaled_attention(x_de.squeeze(0).transpose(1,0), x)
            x_final = x_final + x_update

            x_de = x_decoder[2]
            x_de = self.map_512_32(x_de.unsqueeze(0).transpose(2,1))
            x_update = scaled_attention(x_de.squeeze(0).transpose(1,0), x)
            x_final = x_final + x_update

            x_de = x_decoder[3]
            x_de = self.map_256_32(x_de.unsqueeze(0).transpose(2,1))
            x_update = scaled_attention(x_de.squeeze(0).transpose(1,0), x)
            x_final = x_final + x_update 

        elif assign_choice == 'dynamic_routing': # since the GPU memeroy, it cannot make sense， thus, this part has not finished debug
            x_de = x_decoder[0]
            x_de0 = self.map_2048_32(x_de.unsqueeze(0).transpose(2,1))            

            x_de = x_decoder[1]
            x_de1 = self.map_1024_32(x_de.unsqueeze(0).transpose(2,1))

            x_de = x_decoder[2]
            x_de2 = self.map_512_32(x_de.unsqueeze(0).transpose(2,1))

            x_de = x_decoder[3]
            x_de3 = self.map_256_32(x_de.unsqueeze(0).transpose(2,1))
            
            # For upsampling 
            N_s, N_t = batch['stack_lengths'][0]
            if st == 0:
                xyz1 = batch['points'][0][:N_s].unsqueeze(0).permute(0,2,1)
                points1 = None 

                xyz2_0 = batch['points'][4][:batch['stack_lengths'][4][0]].unsqueeze(0).permute(0,2,1)
                points2_0 = x_de0
                x_de_dense0 = self.FP_upsample(xyz1, xyz2_0, points1, points2_0).permute(0,2,1) #1,N,32

                xyz2_1 = batch['points'][3][:batch['stack_lengths'][3][0]].unsqueeze(0).permute(0,2,1)
                points2_1 = x_de1
                x_de_dense1 = self.FP_upsample(xyz1, xyz2_1, points1, points2_1).permute(0,2,1)

                xyz2_2 = batch['points'][2][:batch['stack_lengths'][2][0]].unsqueeze(0).permute(0,2,1)
                points2_2 = x_de2
                x_de_dense2 = self.FP_upsample(xyz1, xyz2_2, points1, points2_2).permute(0,2,1)

                xyz2_3 = batch['points'][1][:batch['stack_lengths'][1][0]].unsqueeze(0).permute(0,2,1)
                points2_3 = x_de3
                x_de_dense3 = self.FP_upsample(xyz1, xyz2_3, points1, points2_3).permute(0,2,1)

            elif st == 1:
                xyz1 = batch['points'][0][N_s:].unsqueeze(0).permute(0,2,1)
                points1 = None 

                xyz2_0 = batch['points'][4][batch['stack_lengths'][4][0]:].unsqueeze(0).permute(0,2,1)
                points2_0 = x_de0
                x_de_dense0 = self.FP_upsample(xyz1, xyz2_0, points1, points2_0).permute(0,2,1) #1,N,32

                xyz2_1 = batch['points'][3][batch['stack_lengths'][3][0]:].unsqueeze(0).permute(0,2,1)
                points2_1 = x_de1
                x_de_dense1 = self.FP_upsample(xyz1, xyz2_1, points1, points2_1).permute(0,2,1)

                xyz2_2 = batch['points'][2][batch['stack_lengths'][2][0]:].unsqueeze(0).permute(0,2,1)
                points2_2 = x_de2
                x_de_dense2 = self.FP_upsample(xyz1, xyz2_2, points1, points2_2).permute(0,2,1)

                xyz2_3 = batch['points'][1][batch['stack_lengths'][1][0]:].unsqueeze(0).permute(0,2,1)
                points2_3 = x_de3
                x_de_dense3 = self.FP_upsample(xyz1, xyz2_3, points1, points2_3).permute(0,2,1)
            #x_de_denses.append(x_de_dense)
            # x_de_dense0 = upsampleing(x_de0.squeeze(0).transpose(1,0), batch, st).unsqueeze(0) # 1,N,32            
            # x_de_dense1 = upsampleing(x_de1.squeeze(0).transpose(1,0), batch, st).unsqueeze(0) # 1,N,32
            # x_de_dense2 = upsampleing(x_de2.squeeze(0).transpose(1,0), batch, st).unsqueeze(0) # 1,N,32
            # x_de_dense3 = upsampleing(x_de3.squeeze(0).transpose(1,0), batch, st).unsqueeze(0) # 1,N,32
            x_de_dense4 = x.unsqueeze(0) # 1,N,32
            #x_de_denses.append(x_de)
            x_de_denses = torch.cat([x_de_dense0,x_de_dense1,x_de_dense2,x_de_dense3,x_de_dense4],dim=0)

            # dr learning 
            feature = x_de_denses.permute(1,0,2) # N,5,D                        
            x_final = self.latent_caps_learner(feature)
            #x_final = DR_learning(x_de_denses)  

        return x_final.squeeze(1)
        
def upsampleing(x_de, batch, i):
    if i == 0:
        N = batch['stack_lengths'][0][0]
        x_upsampled = closest_pool(x_de, batch['upsamples'][0][:N])
    if i == 1:
        N = batch['stack_lengths'][0][0]
        x_upsampled = closest_pool(x_de, batch['upsamples'][0][N:])
    else:
        print('error')
    return x_upsampled

def DR(x_de_denses, x):  
    def squash_dr(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

    xs = np.stack(x_de_denses) # 4,N,D
    u_hat = np.concatnate([xs, x], axis=0).transpose(1,0,2) # N,5,D
    latent_caps_size = 1
    prim_caps_size = u_hat.shape[-1]
    b_ij = Variable(torch.zeros(x.size(0), latent_caps_size, prim_caps_size)).cuda()
    num_iterations = 3
    for iteration in range(num_iterations):
        c_ij = F.softmax(b_ij, 1)
        if iteration == num_iterations - 1:
            v_j = squash_dr(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
        else:
            v_j = squash_dr(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            b_ij = b_ij + torch.sum(v_j * u_hat, dim=-1)
    return v_j.squeeze(-2).unsqueeze(-2)

def DR_learning(x_de_denses):# 5,N,D
    feature = x_de_denses.permute(1,0,2) # N,5,D
    _, prim_caps_size, prim_vec_size = feature.shape
    latent_vec_size = prim_vec_size
    latent_caps_learner = LatentCapsLayer(1, prim_caps_size, prim_vec_size, latent_vec_size)
    x_final = latent_caps_learner(feature)
    return x_final.unsqueeze(-2)

class MLP2048_32(nn.Module):
    def __init__(self):
        super(MLP2048_32, self).__init__()
        self.conv1 = nn.Conv1d(2048, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 32, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class MLP1024_32(nn.Module):
    def __init__(self):
        super(MLP1024_32, self).__init__()
        self.conv1 = nn.Conv1d(2048, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 32, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(32)

    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class MLP512_32(nn.Module):
    def __init__(self):
        super(MLP512_32, self).__init__()
        self.conv1 = nn.Conv1d(2048, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 32, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(32)

    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class MLP256_32(nn.Module):
    def __init__(self):
        super(MLP256_32, self).__init__()
        self.conv1 = nn.Conv1d(2048, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 32, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(32)

    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class MLP2048_16(nn.Module):
    def __init__(self):
        super(MLP2048_16, self).__init__()
        self.conv1 = nn.Conv1d(2048, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 16, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class MLP1024_16(nn.Module):
    def __init__(self):
        super(MLP1024_16, self).__init__()
        self.conv1 = nn.Conv1d(2048, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 16, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(16)

    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class MLP512_16(nn.Module):
    def __init__(self):
        super(MLP512_16, self).__init__()
        self.conv1 = nn.Conv1d(2048, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 16, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(16)

    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class MLP256_16(nn.Module):
    def __init__(self):
        super(MLP256_16, self).__init__()
        self.conv1 = nn.Conv1d(2048, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.conv4 = nn.Conv1d(256, 16, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(16)

    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

### function : corss by scaled attention metric ### 
def scaled_attention(x, y): # x: N1,D; y: N2,D; -> N2,D
    d_k = x.size(-1)
    scores = torch.matmul(x, y.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn.transpose(-2,-1), x) #+ y


### class : cross by transformer ###
class Transformer(nn.Module):
    def __init__(self, args, layer):
        super(Transformer, self).__init__()
        self.layer = layer
        self.emb_dims = args.emb_dims[layer]
        self.N = 1
        self.dropout = 0.0
        self.ff_dims = args.ff_dims[layer]
        self.n_heads = 4
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, batch, feature_encoder):
        #L = len(feature_encoder) 
        #pair_embeddings = []
        #for i in range(L):
        N_src, N_tgt = batch['stack_lengths'][4-self.layer]
        src_f = feature_encoder[self.layer][:N_src,:].unsqueeze(0) # B,N,D
        tgt_f = feature_encoder[self.layer][N_src:,:].unsqueeze(0)

        tgt_embedding = self.model(src_f, tgt_f, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt_f, src_f, None, None).transpose(2, 1).contiguous()
        pair_embedding = torch.cat([src_embedding, tgt_embedding],dim=-1).squeeze(0)
            #pair_embeddings.append(pair_embedding)
        return pair_embedding.transpose(1,0)

### class : learn the scores of key points

class CapNet(nn.Module):
    def __init__(self, args):
        super(CapNet, self).__init__()
        #self.emb_dims = args.emb_dims
        self.latent_caps_layer = LatentCapsLayer(args.latent_caps_size, args.prim_caps_size, args.prim_vec_size, args.latent_vec_size)

    def forward(self, input, feature):
        neighbor_index = input['select_cap']
        latent_capsule_summary = []
        for i in tqdm(range(feature.shape[0])):
            feat = gather(feature, neighbor_index[i].long()).unsqueeze(0) # 1,N,C
            latent_capsules_i = self.latent_caps_layer(feat)
            latent_capsule_summary.append([latent_capsules_i])
        latent_capsule_summary = torch.stack(latent_capsule_summary)

        N_src0, N_tgt0 = input['stack_lengths'][0]

        init_feature_src = feature[:N_src0,:] # N,F
        init_feature_tgt = feature[N_src0:,:] # N,F
        index_src = input['select_cap'][:N_src0,:] # N,c
        index_tgt = input['select_cap'][N_src0:,:] # N,c

        batch_feat_src = gather(init_feature_src, index_src.long())#.permute(0,2,1) # N,c,F -> N, F, c
        batch_feat_tgt = gather(init_feature_tgt, index_tgt.long())#.permute(0,2,1)
        N,n,F = batch_feat_src.shape
        batch_feat = torch.zeros(N,50,n,F,1)
        for i in range(50):
            batch_feat[:,i,:,:,0] = batch_feat_src.clone()

        latent_capsules_src = self.latent_caps_layer(batch_feat)
        batch_feat = torch.cat((batch_feat_src, batch_feat_tgt),dim=0)
        
        latent_capsules_src = self.latent_caps_layer(batch_feat)
        N,n,F = batch_feat.shape
        B = int(N / 10)+1
        temp = torch.zeros(1,n,F)
        for i in range(B):
            if i < B-1:
                batch_feat_i = batch_feat[i*10:(i+1)*10]
                latent_capsules_i = self.latent_caps_layer(batch_feat_i)
            elif i==B-1:
                batch_feat_i = batch_feat[i*10:]
                latent_capsules_i = self.latent_caps_layer(batch_feat_i)
            temp = torch.cat((temp,latent_capsules_i),dim=0)
        latent_capsules = temp[1:]

        
        # latent_capsules_src = self.latent_caps_layer(batch_feat_src)
        # latent_capsules_tgt = self.latent_caps_layer(batch_feat_tgt)

        # latent_capsules = torch.cat((latent_capsules_src, latent_capsules_tgt),dim=1)
        return latent_capsules.squeeze(0), torch.norm(latent_capsules,dim=-1).permute(1,0)

class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=16, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size)).cuda() #LC, PC, LV, PV # 50,50,32,32

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class DynamicScores(nn.Module):
    def __init__(self, args):
        super(DynamicScores, self).__init__()

    def forward(self, inputs, features):
        neighbor = inputs['neighbors'][0]
        first_pcd_length, second_pcd_length = inputs['stack_lengths'][0]
        first_pcd_indices = torch.arange(first_pcd_length)
        second_pcd_indices = torch.arange(first_pcd_length, first_pcd_length+second_pcd_length)

        # add a fake point in the last row for shadown neighbors
        shadow_features = torch.zeros_like(features[:1,:])
        features = torch.cat([features, shadow_features], dim=0)
        shadow_neighbor = torch.ones_like(neighbor[:1,:]) * (first_pcd_length + second_pcd_length)
        neighbor = torch.cat([neighbor, shadow_neighbor], dim=0)
        features = features / (torch.max(features) + 1e-6)

        # local max score (saliency socre)
        neighbor_features = features[neighbor, :] # [n_points, n_neighbors, 64]
        neighbor_features_sum = torch.sum(neighbor_features, dim=-1) # [n_points, n_neighbors]
        neighbor_num = (neighbor_features_sum != 0).sum(dim=-1, keepdims=True) # [n_points, 1]
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        mean_features = torch.sum(neighbor_features, dim=1) / neighbor_num # [n_points, 64]
        local_max_score = F.softplus(features - mean_features) # [n_points, 64]

        # calculate the depth-wise max score
        depth_wise_max = torch.max(features, dim=1, keepdims=True)[0] # [n_points, 1]
        depth_wise_max_score = features / (1e-6 + depth_wise_max) # [n_points, 64]
        all_scores = local_max_score * depth_wise_max_score 

        # use the max score among channel to be the score of a single point 
        scores = torch.max(all_scores, dim=1, keepdims=True)[0] # [n_points, 1]
        if self.training is False:
            local_max = torch.max(neighbor_features, dim=1)[0]
            is_local_max = (features == local_max)
            detected = torch.max(is_local_max.float(), dim=1, keepdims=True)[0]
            scores = scores * detected
        
        return scores[:-1,:]

######## Others  #########
# for transformer 
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn
