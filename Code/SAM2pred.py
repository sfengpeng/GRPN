from segment_anything import sam_model_registry
import os
import torch
from torch import nn
import cv2
import torch.nn.functional as F
import numpy as np


from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

def entropy_fuse(SAM_logits, model_logits):
    SAM_logits = torch.sigmoid(SAM_logits)
    SAM_logits = torch.cat([1-SAM_logits, SAM_logits], dim = 1)
    model_logits = model_logits.softmax(dim = 1)

    # calculate entropy
    def calculate_entropy(pred):
        epsilon = 1e-10  # Small value to avoid log(0)
        foreground_pred = pred[:, 1, :, :]  # Extract the foreground predictions
        background_pred = pred[:, 0, :, :]  # Extract the background predictions
        
        # Calculate entropy
        entropy = - (foreground_pred * torch.log(foreground_pred + epsilon) + 
                    background_pred * torch.log(background_pred + epsilon))
        
        return entropy
    
    entropy_sam = calculate_entropy(SAM_logits).unsqueeze(1) 
    entropy_model = calculate_entropy(model_logits).unsqueeze(1) 

    total_entropy = entropy_sam + entropy_model

    weight_sam = entropy_sam / total_entropy
    weight_model = entropy_model / total_entropy

    return weight_model * SAM_logits + weight_sam* model_logits




def sum_fuse(SAM_logits, model_logits):
    SAM_logits = torch.sigmoid(SAM_logits)
    SAM_logits = torch.cat([1-SAM_logits, SAM_logits], dim = 1)
    model_logits = model_logits.softmax(dim = 1)

    return F.softmax(model_logits + SAM_logits, dim = 1)



def coff_fuse(SAM_logits, model_logits, alpha):
    SAM_logits = torch.sigmoid(SAM_logits)
    SAM_logits = torch.cat([1-SAM_logits, SAM_logits], dim = 1)
    model_logits = model_logits.softmax(dim = 1)

    return alpha * SAM_logits + (1 - alpha) * model_logits



def union_fuse(SAM_logits, model_logits):

     SAM_logits = SAM_logits.squeeze(1) # 
     model_logits =model_logits.argmax(dim = 1) # 
     pred = torch.logical_or(SAM_logits, model_logits) #并集
     return pred



def xor_fuse(SAM_logits, model_logits):

     SAM_logits = SAM_logits.squeeze(1) # 
     model_logits =model_logits.argmax(dim = 1) # 
     pred = torch.logical_xor(SAM_logits, model_logits) #并集
     return pred



def ot_fuse(SAM_logits, model_logits):
    SAM_logits = torch.sigmoid(SAM_logits)
    SAM_logits = torch.cat([1-SAM_logits, SAM_logits], dim = 1)
    model_logits = model_logits.softmax(dim = 1)

class SAM_pred(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.sam_model = sam_model_registry[args.sam_type](args.ckpt)
        self.sam_model.eval()
        self.point = args.point 
        self.negative_point = args.negative_point
        self.positive_point = args.positive_point
        self.args = args

    def forward_img_encoder(self, query_img):
        query_img = F.interpolate(query_img, (1024,1024), mode='bilinear', align_corners=True)

        with torch.no_grad():
            query_feats = self.sam_model.image_encoder(query_img)

        return query_feats
    

    def get_center_of_binary_mask(mask):
        B, H, W = mask.shape
        # 确保掩码是二值的
        assert mask.max() <= 1 and mask.min() >= 0, "Mask should be binary (0 or 1)."

        # 初始化中心位置张量
        centers = torch.zeros((B, 2), dtype=torch.int64)

        for b in range(B):
            binary_mask = mask[b].bool()
            if binary_mask.sum() > 0:
                # 计算掩码的矩
                indices = torch.nonzero(binary_mask, as_tuple=False)
                cY = indices[:, 0].float().mean().round().long()
                cX = indices[:, 1].float().mean().round().long()
                centers[b] = torch.tensor([cX, cY])
            else:
                raise ValueError(f"The mask at batch index {b} has no non-zero pixels.")

        return centers

    
    def get_feat_from_np(self, query_img, query_name, protos):
        np_feat_path = '/root/paddlejob/workspace/env_run/vrp_sam/feats_np/coco/'
        if not os.path.exists(np_feat_path): os.makedirs(np_feat_path)
        files_name = os.listdir(np_feat_path)
        query_feat_list = []
        for idx, name in enumerate(query_name):
            if '/root' in name:
                name = os.path.splitext(name.split('/')[-1])[0]
                
            if name + '.npy' not in files_name:
                query_feats_np = self.forward_img_encoder(query_img[idx, :, :, :].unsqueeze(0))
                query_feat_list.append(query_feats_np)
                query_feats_np = query_feats_np.detach().cpu().numpy()
                np.save(np_feat_path + name + '.npy', query_feats_np)
            else:
                sub_query_feat = torch.from_numpy(np.load(np_feat_path + name + '.npy')).to(protos.device)
                query_feat_list.append(sub_query_feat)
                del sub_query_feat
        query_feats_np = torch.cat(query_feat_list, dim=0)
        return query_feats_np

    def get_pormpt(self, protos, points_mask=None):
        if points_mask is not None :
            point_mask = points_mask

            postivate_pos = (point_mask.squeeze(0).nonzero().unsqueeze(0) + 0.5) * 64 -0.5
            postivate_pos = postivate_pos[:,:,[1,0]]
            point_label = torch.ones(postivate_pos.shape[0], postivate_pos.shape[1]).to(postivate_pos.device)
            point_prompt = (postivate_pos, point_label)
        else:
            point_prompt = None
        protos = protos
        return  protos, point_prompt

    def forward_prompt_encoder(self, points=None, boxes=None, protos=None, masks=None):
        q_sparse_em, q_dense_em = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                protos=protos,
                masks=None)
        return q_sparse_em, q_dense_em
    
    def forward_mask_decoder(self, query_feats, q_sparse_em, q_dense_em, ori_size, protos = None, attn_sim=None):
        # if protos is not None: 
        #     protos = torch.mean(protos, dim = 1, keepdim=True)
        
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=query_feats,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=q_sparse_em,
                dense_prompt_embeddings=q_dense_em,
                protos = protos,
                attn_sim = attn_sim,
                multimask_output=False)
        low_masks = F.interpolate(low_res_masks, size=ori_size, mode='bilinear', align_corners=True)
            
        # from torch.nn.functional import threshold, normalize

        # binary_mask = normalize(threshold(low_masks, 0.0, 0))
        binary_mask = torch.where(low_masks > 0, 1, 0)
        return low_masks, binary_mask
    
    
    def forward(self, query_img, prediction, protos, origin_pred = None, points_mask=None, enhance_q = None):
        # B,C, h, w = query_img.shape
        h, w = 400, 400
        positive_vis, ne_vis = torch.rand([1,2], requires_grad=False), torch.rand(2,3, requires_grad=False)
        
        coords, labels, positive_vis, ne_vis, vector = self.point_mask_slic(origin_pred, enhance_q) # best 
        #coords, labels = self.ponit_selection(origin_pred)
        #box = self.get_box(origin_pred)

        with torch.no_grad():
        #     #-------------save_sam_img_feat------------------------- # save_sam_img_feat
             query_feats = self.forward_img_encoder(query_img)

        #     #query_feats = self.get_feat_from_np(query_img, query_name, protos)

        q_sparse_em, q_dense_em = self.forward_prompt_encoder(
                points=(coords, labels),
                boxes=None,
                protos=None,
                masks=None)
            
        low_masks, binary_mask = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em, (h, w), protos = None, attn_sim=None)


        # q_sparse_em, q_dense_em = self.forward_prompt_encoder(
        #         points=(coords, labels),
        #         boxes=None,
        #         protos=None,
        #         masks=low_masks)
            
        # low_masks, binary_mask = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em, (h, w), protos = None, attn_sim=None)


        type =self.args.fuse_method

        if 'entropy' == type:
            pred = entropy_fuse(low_masks, prediction)
        elif 'sum' == type:
            pred = sum_fuse(low_masks, prediction)
        elif 'coff' == type:
            pred = coff_fuse(low_masks,prediction, self.args.alpha)
        elif 'union' == type:
            pred = union_fuse(binary_mask, prediction)
        elif 'xor' == type:
            pred = xor_fuse(binary_mask, prediction)
        # if type not in ('xor', 'union'):
        #     pred = torch.argmax(pred, dim = 1)

        return pred, low_masks, positive_vis, ne_vis
    

    def ponit_selection(self, pred):
        # pred has shape [B, 2, H, W], H, W are origin imgae size
        pred = F.softmax(pred, dim = 1)[:, 1, :, :] # foreground prediction
        b, h, w = pred.shape

        topk = self.point

        topk_xy_list = []
        topk_label_list = []
        last_xy_list = []
        last_label_list = []

        for idx in range(b):
            pred_idx = pred[idx, :, :]
            topk_xy = pred_idx.flatten(0).topk(topk)[1]
            topk_x = (topk_xy // h).unsqueeze(0)
            topk_y = (topk_xy - topk_x * h)
            topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0) # 
            topk_label = torch.ones(topk, dtype=torch.int)

            topk_xy_list.append(topk_xy)
            topk_label_list.append(topk_label)
                
            # Top-last point selection
            last_xy = pred_idx.flatten(0).topk(topk, largest=False)[1]
            last_x = (last_xy // h).unsqueeze(0)
            last_y = (last_xy - last_x * h)
            last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
            last_label = torch.zeros(topk, dtype=torch.int) # 视为背景
            
            last_xy_list.append(last_xy)
            last_label_list.append(last_label)
            # del last_label, last_xy, topk_label, topk_xy

        topk_label_list = torch.stack(topk_label_list, dim = 0)
        last_label_list = torch.stack(last_label_list, dim = 0)
        topk_xy_list = torch.stack(topk_xy_list, dim = 0)
        last_xy_list = torch.stack(last_xy_list, dim = 0)

        coords = torch.cat([topk_xy_list,last_xy_list], dim = 1)
        labels = torch.cat([topk_label_list,last_label_list], dim = 1)
        coords = self.transform(coords, h, w)
        return coords, labels
    

    def select_all(self, pred):
        pred = pred.argmax(dim=1)
        b, h, w = pred.shape

        xy_list = []
        label_list = []

        for idx in range(b):
            pred_idx = pred[idx, :, :]

            # 将 pred_idx 展平为一维
            flat_pred_idx = pred_idx.flatten()
            
            # 找到值为0的索引
            zero_indices = torch.nonzero(flat_pred_idx == 0, as_tuple=False).squeeze(1)
            zero_x = zero_indices // w
            zero_y = zero_indices % w
            zero_xy = torch.stack((zero_y, zero_x), dim=1)
            zero_label = torch.zeros(zero_xy.size(0), dtype=torch.int)  # 视为背景
            
            # 找到值为1的索引
            one_indices = torch.nonzero(flat_pred_idx == 1, as_tuple=False).squeeze(1)
            one_x = one_indices // w
            one_y = one_indices % w
            one_xy = torch.stack((one_y, one_x), dim=1)
            one_label = torch.ones(one_xy.size(0), dtype=torch.int)  # 视为前景
            
            xy_list.append(torch.cat([zero_xy, one_xy], dim = 0))
            label_list.append(torch.cat([zero_label, one_label], dim = 0))

            del zero_xy, one_xy, zero_label, one_label


        coords = torch.stack(xy_list, dim = 0)
        labels = torch.stack(label_list, dim = 0)
        coords = self.transform(coords, h, w)
        return coords, labels
    

    
    def transform(self, coords, old_h = 400, old_w = 400, new_h = 1024, new_w = 1024):
        # coords has shape [B x N x 2]
        coords = coords.float()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    


    def get_box(self, pred):
        # pred has shape B x 2 X h x w
        pred = pred.argmax(dim = 1)
        b, h, w = pred.shape

        out = []
        for i in range(b):
            pred_idx = pred[i, :, :]
            x, y = torch.nonzero(pred_idx, as_tuple=True)

            # 计算边界框
            x_min = x.min().item()
            x_max = x.max().item()
            y_min = y.min().item()
            y_max = y.max().item()

            input_box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.int)
            out.append(input_box)

        boxes = torch.stack(out, dim = 0)

        boxes = self.transform(boxes.reshape(-1, 2, 2), h, w)
        return boxes.reshape(-1, 4)
    


    def mask_slic(self, pred, point_num, avg_sp_area=100):
        '''
        :param mask: the RoI region to do clustering, torch tensor: H x W
        :param down_stride: downsampled stride for RoI region
        :param max_num_sp: the maximum number of superpixels
        :return: segments: the coordinates of the initial seed, max_num_sp x 2
        '''
        assert point_num >= 0
        mask = pred
        h, w = mask.shape
        max_num_sp = point_num

        segments_x = np.zeros(max_num_sp, dtype=np.int64)
        segments_y = np.zeros(max_num_sp, dtype=np.int64)

        m_np = mask.cpu().numpy()
        down_size = (h, w)
        m_np_down = m_np

        nz = np.nonzero(m_np_down) # 找到所有的非零坐标下标 x坐标，y坐标
        # After transform, there may be no nonzero in the label
        if len(nz[0]) != 0: # 区域内存在掩码区域

            p = [np.min(nz[0]), np.min(nz[1])] # 左上角
            pend = [np.max(nz[0]), np.max(nz[1])] # 右下角

            # cropping to bounding box around ROI
            m_np_roi = np.copy(m_np_down)[p[0]:pend[0] + 1, p[1]:pend[1] + 1] # 感兴趣的区域
            num_sp = max_num_sp

            # num_sp is adaptive, based on the area of support mask
            # mask_area = (m_np_roi == 1).sum()
            # num_sp = int(min((np.array(mask_area) / avg_sp_area).round(), max_num_sp)) # 找到需要设置的num_sp数量

        else:
            num_sp = 0

        if (num_sp != 0) and (num_sp != 1):
            for i in range(num_sp):

                # n seeds are placed as far as possible from every other seed and the edge.

                # STEP 1:  conduct Distance Transform and choose the maximum point
                dtrans = distance_transform_edt(m_np_roi) #  
                dtrans = gaussian_filter(dtrans, sigma=0.1)

                coords1 = np.nonzero(dtrans == np.max(dtrans))
                segments_x[i] = coords1[0][0]
                segments_y[i] = coords1[1][0]

                # STEP 2:  set the point to False and repeat Step 1
                m_np_roi[segments_x[i], segments_y[i]] = False
                segments_x[i] += p[0]
                segments_y[i] += p[1]

        segments = np.concatenate([segments_y[..., np.newaxis], segments_x[..., np.newaxis]], axis=1)  # max_num_sp x 2
        segments = torch.from_numpy(segments)
        segments = segments.to(pred.device)

        return segments
    


    def get_sam_predict(self, fg_mask, bg_mask, feature):
        """SSP refine 

        Args:
            fg_mask (torch.tensor: H X W): 二值化的前景掩码
            bg_mask (torch.tensor: H X W): 二值化的背景掩码
            feature (torch.tensor: H X W): 经过SAM Image Encoder 提取的support或者query特征。 
        """
        h, w = fg_mask.shape

        feature = feature.unsqueeze(0)

        fg_num = torch.sum(fg_mask).item()
        bg_num = torch.sum(bg_mask).item()

        # 求最终的点数量
        fg_num = min(fg_num, self.positive_point)
        bg_num = min(bg_num, self.negative_point)

        seg_fg = self.mask_slic(fg_mask, fg_num)
        seg_bg = self.mask_slic(bg_mask, bg_num)

        fg_label = torch.ones(fg_num, dtype=torch.int)
        bg_label = torch.zeros(bg_num, dtype=torch.int)

        coords = torch.cat([seg_fg, seg_bg], dim = 0).unsqueeze(0) # 1 x N X 2
        labels = torch.cat([fg_label, bg_label], dim = 0).unsqueeze(0) # 1 x N
        labels = labels.to(fg_mask.device)

        coords = self.transform(coords, 50, 50)


        q_sparse_em, q_dense_em = self.forward_prompt_encoder(
                points=(coords, labels),
                boxes=None,
                protos=None,
                masks=None)
            
        low_masks1, binary_mask1 = self.forward_mask_decoder(feature, q_sparse_em, q_dense_em, (h, w), protos = None, attn_sim=None)

        if self.args.second_refine:
            q_sparse_em, q_dense_em = self.forward_prompt_encoder(
                points=(coords, labels),
                boxes=None,
                protos=None,
                masks=low_masks1)
            
            low_masks1, binary_mask1 = self.forward_mask_decoder(feature, q_sparse_em, q_dense_em, (h, w), protos = None, attn_sim=None)

       
        return low_masks1
    

    def point_mask_slic(self, pred, enhance_q):
        b = pred.shape[0]
        coords = []
        labels = []

        positive_vis = []
        negative_vis = []

        
        for i in range(b):
            pred_i = pred[i, :, :, :]
            # positive
            seg_p= self.mask_slic(pred_i.argmax(dim = 0), self.positive_point)
            # negatice
            
            seg_n = self.mask_slic(pred_i.flip(dims=[0]).argmax(dim = 0), self.negative_point)
            
            M = seg_p.shape[0] 
            N = seg_n.shape[0]
            label_p = torch.ones(M, dtype=torch.int)
            label_n = torch.zeros(N, dtype=torch.int)

            coords.append(torch.cat([seg_p, seg_n], dim=0))
            #coords.append(seg_p)
            labels.append(torch.cat([label_p, label_n], dim = 0))
            #labels.append(label_p)
            positive_vis.append(seg_p)
           
            negative_vis.append(seg_n)

        coords = torch.stack(coords, dim = 0) # B X N X 2
        labels = torch.stack(labels, dim = 0)

        # 
        x_coords = coords[:, :, 0]
        y_coords = coords[:, :, 1]

        # 扩展批量维度
        batch_indices = torch.arange(b).view(b, 1).expand(b, coords.shape[1])

        # 使用高级索引从特征图中提取特征向量
        # 形状：B x N x C
        extracted_features = enhance_q[batch_indices, :, y_coords, x_coords] # B X N X C 记得改回来

        labels = labels.to(pred.device)
        coords = self.transform(coords, 50, 50)
        positive_vis = torch.stack(positive_vis, dim = 0)
        negative_vis = torch.stack(negative_vis, dim = 0)
        positive_vis = self.transform(positive_vis, 50, 50, 400, 400)
        negative_vis = self.transform(negative_vis, 50, 50, 400, 400)
        return coords, labels, positive_vis, negative_vis, extracted_features#
    




    


    