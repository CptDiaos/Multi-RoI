from dataclasses import astuple
from difflib import diff_bytes
from os import device_encoding
import sys
sys.path.append('../')
import numpy as np
# from serialization import load_model
from matplotlib import cm as mpl_cm, colors as mpl_colors
import cv2
import os
from tqdm import tqdm
import imageio
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import torch
from torch.nn.functional import interpolate
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.renderer import (
    BlendParams,
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardFlatShader,
    SoftPhongShader,
    SoftSilhouetteShader
)
from pytorch3d.utils import cameras_from_opencv_projection
import json
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa, get_affine_transform, affine_transform






def _construct_rotation_matrix(rot, axis=2, size=3):
    """Construct the in-plane rotation matrix.
    Args:
        rot (float): Rotation angle (degree).
        size (int): The size of the rotation matrix.
            Candidate Values: 2, 3. Defaults to 3.
    Returns:
        rot_mat (np.ndarray([size, size]): Rotation matrix.
    """
    rot_mat = np.eye(size, dtype=np.float32)
    if rot != 0:
        rot_rad = np.deg2rad(rot)
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        # print(sn,cs)
        if axis == 0:
            rot_mat[1, 1:] = [cs, -sn]
            rot_mat[2, 1:] = [sn, cs]
        elif axis == 1:
            rot_mat[0, [0, 2]] = [cs, sn]
            rot_mat[2, [0, 2]] = [-sn, cs]
        else:
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
    return rot_mat




def rotate_smpl_pose(pose, rot, axis=2):
    """Rotate SMPL pose parameters.
    SMPL (https://smpl.is.tue.mpg.de/) is a 3D
    human model.
    Args:
        pose (np.ndarray([72])): SMPL pose parameters
        rot (float): Rotation angle (degree).
        axis: x:0, y:1, z:2
    Returns:
        pose_rotated
    """
    pose_rotated = pose.copy()
    if rot != 0:
        rot_mat = _construct_rotation_matrix(-rot, axis)
        orient = pose[:3]
        # find the rotation of the body in camera frame
        per_rdg, _ = cv2.Rodrigues(orient.astype(np.float32))
        # apply the global rotation to the global orientation
        res_rot, _ = cv2.Rodrigues(np.dot(rot_mat, per_rdg))
        pose_rotated[:3] = (res_rot.T)[0]
    return pose_rotated





def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)

    for part_idx, (k, v) in enumerate(part_segm.items()):
        vertex_labels[v] = part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 3))
    # vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

    return vertex_colors





# def render_smpl(vertices, faces, batch_size, cam_new, trans_z, img_size, scale, device, focal_length, num_view=4):
#     new_imgs_all = []
#     bbox_infos_all = []
#     faces = torch.from_numpy((faces.astype('int32'))).float()
#     for B in range(batch_size):
#         h_np, w_np = img_size[B][0].cpu().numpy().item(), img_size[B][1].cpu().numpy().item()
#         tz = trans_z[B: B+1].repeat(num_view, 1)
#         # print(tz.shape)
#         batch_vertices = vertices[B]
#         batch_img_size = img_size[B: B+1]
#         focal_np = np.sqrt(h_np ** 2 + w_np ** 2)
#         intrinsic_mtx = np.array([[focal_np, 0, w_np/2, 0],
#                                 [0, focal_np, h_np/2, 0],
#                                 [0,     0,   1, 0]])
#         cali_mtx = torch.from_numpy(intrinsic_mtx.astype('float32')).unsqueeze(0).to(device)

#         part_segm = json.load(open('smpl_vert_6segmentation.json'))
#         vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[1])
#         # vertex_colors = np.zeros_like(m)*1.
#         verts_rgb = torch.from_numpy(vertex_colors).unsqueeze(0).float().to(device)
#         raster_settings = RasterizationSettings(
#             image_size=(h_np, w_np), 
#             blur_radius=0.0, 
#             faces_per_pixel=1, 
#             cull_backfaces=True,
#             bin_size=0
#         )
#         # mtx_x = _construct_rotation_matrix(0, axis=0)
#         # mtx_y = _construct_rotation_matrix(0, axis=1)
#         # mtx_all = np.dot(mtx_y, mtx_x).reshape(1, 3, 3)
#         mtx_all = np.eye(3)
#         rt = torch.from_numpy(mtx_all).unsqueeze(0).repeat(num_view, 1, 1).float().to(device)
#         # t = torch.from_numpy(np.array([[0, 0, 7.], [0, 1, 7.], [0, -1, 7.], [-1, -1, 7.], [-1, 0, 7.], [-1, 1, 7.], [1, -1, 7.], [1, 0, 7.], [1, 1, 7.]])).float().to(device)
#         # t = torch.from_numpy(np.array([[-1, 1, 7.], [1, -1, 7.], [1, 1, 7.], [-1, -1, 7.]])).float().to(device)
#         t = torch.cat((cam_new[B], tz), -1)
#         # print(t)

#         cameras = cameras_from_opencv_projection(rt, t, cali_mtx, batch_img_size.to(dtype=torch.int64))
#         lights = PointLights(device=device, specular_color=((0., 0., 0.,),), location=[[0.0, 0.0, -10.0]])
#         renderer = MeshRenderer(
#                                 rasterizer=MeshRasterizer(
#                                     cameras=cameras, 
#                                     raster_settings=raster_settings
#                                 ),
#                                 shader=SoftPhongShader(
#                                     device=device, 
#                                     cameras=cameras,
#                                     lights=lights
#                                 )
#                                 )
#         verts = batch_vertices
#         # faces = faces.verts_idx
#         # verts_rgb = torch.ones_like(verts)[None].to(device)
#         textures = Textures(verts_rgb=verts_rgb)
#         smpl_mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures).extend(num_view)
#         images = renderer(smpl_mesh)
#         # images = images[..., [2, 1, 0]].cpu().detach().numpy()* 255.
#         images = images[..., :3] * 255.
#         # print(images)
#         # print(images.shape)
#         # print(images.dtype)
#         bbox_infos = []
#         new_imgs = []
#         for V in range(num_view):
#             # print(images[V].shape)
#             blued_ind = torch.where((images[V] < 255.))
#             clamped_ind_y = torch.clamp(blued_ind[0], min=0, max=h_np)
#             clamped_ind_x = torch.clamp(blued_ind[1], min=0, max=w_np)
#             if(clamped_ind_x.numel() == 0 or clamped_ind_y.numel() == 0):
#                 new_imgs.append(torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device))
#                 bbox_infos.append(torch.full((1, 3), -10, dtype=torch.float32, device=device))
#                 print('Invalid indices!', blued_ind[0], blued_ind[1])
#                 continue
#             x0 = torch.min(clamped_ind_x)
#             y0 = torch.min(clamped_ind_y)
#             x1 = torch.max(clamped_ind_x)
#             y1 = torch.max(clamped_ind_y)
#             # print(x0, y0, x1, y1)

#             center = [(x0+x1)/2, (y0+y1)/2]
#             scale = 1.1*max(x1-x0, y1-y0) / 200.
#             b = (scale * 200).int()
#             xa = (center[0] - b / 2).int()
#             xb = (center[0] + b / 2).int()
#             ya = (center[1] - b / 2).int()
#             yb = (center[1] + b / 2).int() 
#             # print(xa, ya, xb, yb)
            
#             new_img = torch.full(((yb - ya).int().detach().cpu().numpy(), (xb - xa).int().detach().cpu().numpy(), 3), 255., device=device)
#             if(new_img[y0-ya:-(yb-y1), x0-xa:-(xb-x1)].shape[0]==0 or new_img[y0-ya:-(yb-y1), x0-xa:-(xb-x1)].shape[1]==0):
#                 print(x0, y0, x1, y1)
#                 print(xa, ya, xb ,yb)
#             new_img[y0-ya:y1-ya, x0-xa:x1-xa] = images[V, y0:y1, x0:x1]
#             new_img = interpolate(new_img.permute(2, 0, 1).unsqueeze(0), (224, 224)) / 255.0
#             bbox_info = torch.stack([center[0] - img_size[B][1] / 2., center[1] - img_size[B][0] / 2., scale]).unsqueeze(0)
#             bbox_info[:, :2] = bbox_info[:, :2] / focal_length[B] * 2.8  # [-1, 1]
#             bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length[B]) / (0.06 * focal_length[B])  # [-1, 1]
#             # print(new_img.shape)
#             # print(bbox_info.shape)
#             new_imgs.append(new_img)
#             bbox_infos.append(bbox_info)
            
#             # cv2.imwrite('/home/lab345/PycharmProjects/SPIN/rendered_img-{}-{}-{}_full.png'.format(t[V][0], t[V][1], t[V][2]), images[V].cpu().detach().numpy())
#             # cv2.imwrite('/home/lab345/PycharmProjects/SPIN/rendered_img-{}-{}-{}.png'.format(t[V][0], t[V][1], t[V][2]), images[V, y0:y1, x0:x1].cpu().detach().numpy())
#         new_imgs=torch.cat(new_imgs, 1)
#         bbox_infos=torch.cat(bbox_infos, 1)
#         # print(new_imgs.shape)
#         # print(bbox_infos.shape)
#         new_imgs_all.append(new_imgs)
#         bbox_infos_all.append(bbox_infos)
#     new_imgs_all=torch.cat(new_imgs_all, 0)
#     bbox_infos_all=torch.cat(bbox_infos_all, 0)
#     # print(new_imgs_all.shape)
#     # print(bbox_infos_all.shape)
#     return new_imgs_all, bbox_infos_all
def render_smpl_vertices(vertices, faces, batch_size, cam_new, trans_z, img_size, device, name, type='gt', num_view=4):
    new_imgs_all = []
    bbox_infos_all = []
    faces = torch.from_numpy((faces.astype('int32'))).float()
    for B in range(batch_size):
        h_np, w_np = img_size[B][0].cpu().numpy().item(), img_size[B][1].cpu().numpy().item()
        tz = trans_z[B: B+1].repeat(num_view, 1)
        # print(tz.shape)
        batch_vertices = vertices[B]
        batch_img_size = img_size[B: B+1]
        focal_np = np.sqrt(h_np ** 2 + w_np ** 2)
        intrinsic_mtx = np.array([[focal_np, 0, w_np/2, 0],
                                [0, focal_np, h_np/2, 0],
                                [0,     0,   1, 0]])
        cali_mtx = torch.from_numpy(intrinsic_mtx.astype('float32')).unsqueeze(0).to(device)

        part_segm = json.load(open('smpl_vert_6segmentation.json'))
        vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[1])
        # vertex_colors = np.zeros_like(m)*1.
        verts_rgb = torch.from_numpy(vertex_colors).unsqueeze(0).float().to(device)
        raster_settings = RasterizationSettings(
            image_size=(h_np, w_np), 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            cull_backfaces=True,
            bin_size=0
        )
        # mtx_x = _construct_rotation_matrix(0, axis=0)
        # mtx_y = _construct_rotation_matrix(0, axis=1)
        # mtx_all = np.dot(mtx_y, mtx_x).reshape(1, 3, 3)
        mtx_all = np.eye(3)
        rt = torch.from_numpy(mtx_all).unsqueeze(0).repeat(num_view, 1, 1).float().to(device)
        # t = torch.from_numpy(np.array([[0, 0, 7.], [0, 1, 7.], [0, -1, 7.], [-1, -1, 7.], [-1, 0, 7.], [-1, 1, 7.], [1, -1, 7.], [1, 0, 7.], [1, 1, 7.]])).float().to(device)
        # t = torch.from_numpy(np.array([[-1, 1, 7.], [1, -1, 7.], [1, 1, 7.], [-1, -1, 7.]])).float().to(device)
        t = torch.cat((cam_new[B], tz), -1)
        # print(t)

        cameras = cameras_from_opencv_projection(rt, t, cali_mtx, batch_img_size.to(dtype=torch.int64))
        lights = PointLights(device=device, specular_color=((0., 0., 0.,),), location=[[0.0, 0.0, -10.0]])
        renderer = MeshRenderer(
                                rasterizer=MeshRasterizer(
                                    cameras=cameras, 
                                    raster_settings=raster_settings
                                ),
                                shader=SoftPhongShader(
                                    device=device, 
                                    cameras=cameras,
                                    lights=lights
                                )
                                )
        verts = batch_vertices
        # faces = faces.verts_idx
        # verts_rgb = torch.ones_like(verts)[None].to(device)
        textures = Textures(verts_rgb=verts_rgb)
        smpl_mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures).extend(num_view)
        images = renderer(smpl_mesh)
        # images = images[..., [2, 1, 0]].cpu().detach().numpy()* 255.
        images = images[..., :3] * 255.
        dir = name[B].split('/')[-1].split('.')[0]
        for V in range(num_view): 
            save_dir = os.path.join('/home/lab345/PycharmProjects/SPIN/data', dir)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite('{}/{}-{:.1f}-{:.1f}-{:.1f}-{}_full.png'.format(save_dir, dir, t[V][0], t[V][1], t[V][2], type), images[V].cpu().detach().numpy())
        # cv2.imwrite('/home/lab345/PycharmProjects/SPIN/rendered_img-{}-{}-{}.png'.format(t[V][0], t[V][1], t[V][2]), images[V, y0:y1, x0:x1].cpu().detach().numpy())
    # print(new_imgs_all.shape)
    # print(bbox_infos_all.shape)
    # return new_imgs_all, bbox_infos_all
    # return images
        



def render_smpl(vertices, vertices_gt, faces, batch_size, cam_ori, cam_new, trans_z, img_size, imgs, scale, device, focal_length, num_view, delta_rot, rot_num=6, offset_num=10, vertices_num=25):
    vertices_bboxes_all = []
    bbox_infos_all = []
    faces = torch.from_numpy((faces.astype('int32'))).float()

    h_np, w_np = img_size[:, 0:1], img_size[:, 1:2] # (3,) (3,)
    tz = trans_z.unsqueeze(1).repeat(1, num_view, 1)  # (3, 4, 1)
    # vertices_3d = torch.cat((vertices, torch.tensor([[[1]]], device=device).repeat(batch_size*rot_num*offset_num, vertices_num, 1)), -1).unsqueeze(1).repeat(1, num_view, 1, 1).reshape(-1, vertices_num ,4)  # (180*4, 25, 4)
    vertices_pred = torch.cat((vertices, torch.tensor([[[1]]], device=device).repeat(batch_size*rot_num*offset_num, vertices_num, 1)), -1).unsqueeze(1).repeat(1, 4, 1, 1)  # (180, 4, 25, 4)
    vertices_gt = torch.cat((vertices_gt.unsqueeze(1).unsqueeze(1).repeat(1, rot_num, offset_num, 1, 1).reshape(-1, vertices_num, 3), torch.tensor([[[1]]], device=device).repeat(batch_size*rot_num*offset_num, vertices_num, 1)), -1).unsqueeze(1)  # (180, 1, 25, 4)
    vertices_3d = torch.cat([vertices_pred, vertices_gt], 1).reshape(-1, vertices_num, 4)  # (180*5, 25, 4)
    focal_np = torch.sqrt(h_np ** 2 + w_np ** 2)
    # print(h_np, w_np)
    # print(vertices_3d)
    
    intrinsic_mtxr0 = torch.cat([focal_np, torch.zeros_like(focal_np, device=0, dtype=torch.float32), w_np/2], -1)
    intrinsic_mtxr1 = torch.cat([torch.zeros_like(focal_np, device=0, dtype=torch.float32), focal_np, h_np/2], -1)
    intrinsic_mtxr2 = torch.tensor([[0, 0, 1]], device=device, dtype=torch.float32).repeat(batch_size, 1)
    intrinsic_mtx = torch.stack([intrinsic_mtxr0, intrinsic_mtxr1, intrinsic_mtxr2], 1).unsqueeze(1).repeat(1, num_view, 1, 1).reshape(-1, 3, 3)
    # print(intrinsic_mtx.shape)
    cams = torch.cat([cam_new, cam_ori.unsqueeze(1)], dim=1)
    t = torch.cat((cams, tz), -1).reshape(-1, 3, 1)
    # print(t.shape)
    trans_t = torch.cat((torch.diag(torch.tensor([1., 1., 1.], device=device)).repeat(batch_size*num_view, 1, 1), t), -1)
    # print(trans_t.shape)
    projection_mtx = torch.einsum('vij,vjk->vik', intrinsic_mtx, trans_t).reshape(batch_size, num_view, 3, 4).unsqueeze(1).repeat(1, rot_num*offset_num, 1, 1, 1).reshape(-1, 3, 4)
    # print(projection_mtx.shape, vertices_3d.shape)
    vertices_reproj = torch.einsum('vij, vkj-> vki', projection_mtx, vertices_3d)
    vertices_reproj = vertices_reproj[...] / vertices_reproj[..., 2:]

    x0, _ = torch.min(vertices_reproj[..., 0], dim=-1)
    y0, _ = torch.min(vertices_reproj[..., 1], dim=-1)
    x1, _ = torch.max(vertices_reproj[..., 0], dim=-1)
    y1, _ = torch.max(vertices_reproj[..., 1], dim=-1)

    x0, y0, x1, y1 = x0.cpu().detach().numpy(), y0.cpu().detach().numpy(), x1.cpu().detach().numpy(), y1.cpu().detach().numpy()

    center = np.array([(x0+x1)/2, (y0+y1)/2])
    bbox_w_h = np.stack([x1 - x0, y1 - y0], 0)
    bbox_b = np.max(bbox_w_h, 0)
    scale = np.array([1.1*bbox_b / 200., 1.1*bbox_b / 200.])
    # print(center.shape, scale.shape)

    crop_trans = []
    for i in range(batch_size*rot_num*offset_num*num_view):
        crop_trans.append(torch.from_numpy(get_affine_transform(center[:, i], scale[:, i], 0, (224, 224))).to(device).float())
    crop_trans = torch.stack(crop_trans, 0)
    vertices_bbox = torch.einsum('vij,vkj->vki', crop_trans,
                                vertices_reproj)
    # image_bbox = dummy_img.index_put((vertices_bbox[:, 1].long(), vertices_bbox[:, 0].long()), torch.tensor([0., 0., 255.], device=device, dtype=torch.float32))
    center_tensor = torch.from_numpy(center).to(device).float().transpose(1, 0)
    scale_tensor = torch.from_numpy(scale)[0].to(device).float()
    # print(center_tensor.shape, scale_tensor.shape)
    focal_length = focal_length.unsqueeze(1).repeat(1, rot_num*offset_num*num_view).reshape(-1)
    # print(focal_length.shape)
    bbox_info = torch.stack([center_tensor[..., 0] - img_size[:, 1:2].repeat(1, rot_num*offset_num*num_view).reshape(-1) / 2., center_tensor[..., 1] - img_size[:, 0:1].repeat(1, rot_num*offset_num*num_view).reshape(-1) / 2., scale_tensor], -1)
    # print(bbox_info.shape)
    bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]
    # print(vertices_bbox.shape, bbox_info.shape)
    vertices_bbox = vertices_bbox.reshape(-1, num_view*vertices_num, 2)
    bbox_info = bbox_info.reshape(-1, num_view*3)
    # # visualize
    # vertices_bbox = vertices_bbox.reshape(batch_size, rot_num, offset_num, num_view, vertices_num, 2)
    # bbox_info = bbox_info.reshape(batch_size, rot_num, offset_num, num_view, 3)
    # vertices_bbox = vertices_bbox.cpu().detach().numpy()
    # for i in range(vertices_bbox.shape[0]):
    #     for j in range(vertices_bbox.shape[1]):
    #         for k in range(vertices_bbox.shape[2]):
    #             for p in range(vertices_bbox.shape[3]):
    #                 dummy_img = np.full((224, 224, 3), 255.)
    #                 for q in range(vertices_bbox.shape[4]):
    #                     point = vertices_bbox[i, j, k, p, q]
    #                     cv2.circle(dummy_img, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)
    #                 cv2.imwrite('/home/lab345/PycharmProjects/SPIN/img_kp/rendered_img-{}-{}-{}-{}.png'.format(i, j, k, p), dummy_img)
    # for i in range(vertices_bbox.shape[0]):
    #     j, k = 3, 5
    #     dummy_img = np.full((224, 224, 3), 255.)
    #     cv2.imwrite('/home/lab345/PycharmProjects/SPIN/img_vis/{}-{}-{}-ori_img.png'.format(i, j, k), imgs[i, [2, 1, 0]].permute(1, 2, 0).cpu().detach().numpy() * 255.0)
    #     for p in range(vertices_bbox.shape[3]):
    #         dummy_img = np.full((224, 224, 3), 255.)
    #         for q in range(vertices_bbox.shape[4]):
    #             point = vertices_bbox[i, j, k, p, q]
    #             cv2.circle(dummy_img, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)
    #         cv2.imwrite('/home/lab345/PycharmProjects/SPIN/img_vis/{}-{}-{}-{}-rendered_img.png'.format(i, j, k, p), dummy_img)

    # print(sss)
    return vertices_bbox, bbox_info
    

    # for B in range(batch_size):
    #     h_np, w_np = img_size[B][0].cpu().numpy().item(), img_size[B][1].cpu().numpy().item()
    #     tz = trans_z[B: B+1].repeat(num_view, 1)
        
    #     batch_vertices = vertices[B].unsqueeze(0).repeat(num_view, 1, 1)
    #     vertices_3d = torch.cat((batch_vertices, torch.tensor([[1]], device=device).unsqueeze(0).repeat(num_view, 25, 1)), -1)
    #     # print(key_points.requires_grad)
    #     batch_img_size = img_size[B: B+1]
    #     focal_np = np.sqrt(h_np ** 2 + w_np ** 2)
    #     intrinsic_mtx = np.array([[focal_np, 0, w_np/2],
    #                             [0, focal_np, h_np/2],
    #                             [0,     0,   1]])
    #     cali_mtx = torch.from_numpy(intrinsic_mtx.astype('float32')).unsqueeze(0).repeat(num_view, 1, 1).to(device)

    #     part_segm = json.load(open('smpl_vert_6segmentation.json'))
    #     # vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[1])
    #     # verts_rgb = torch.from_numpy(vertex_colors).unsqueeze(0).float().to(device)

    #     mtx_all = np.eye(3)
    #     rt = torch.from_numpy(mtx_all).unsqueeze(0).repeat(num_view, 1, 1).float().to(device)
    #     t = torch.cat((cam_new[B], tz), -1).unsqueeze(-1)
    #     trans_t = torch.cat((torch.diag(torch.tensor([1., 1., 1.], device=device)).repeat(num_view, 1, 1), t), -1)
    #     # print(trans_t.requires_grad)  
    #     pose = torch.cat((torch.einsum('vij,vjk->vik',rt, trans_t), torch.tensor([0., 0., 0., 1.], device=device).reshape(1, 1, 4).repeat(num_view, 1, 1)), axis=1)
    #     # print(pose.requires_grad)
    #     # print(t)
    #     images = torch.ones((num_view, h_np, w_np, 3), device=device, dtype=torch.float32, requires_grad=True) * 255.
    #     # print(images.requires_grad)
    #     projection_mtx = torch.einsum('vij,vjk->vik', cali_mtx, trans_t)
    #     vertices_reproj = torch.einsum('vij, vkj-> vki', projection_mtx, vertices_3d)
    #     vertices_reproj = vertices_reproj[...] / vertices_reproj[..., 2:]
        
    #     # print(images)
    #     # print(images.shape)
    #     # print(images.dtype)
    #     bbox_infos = []
    #     vertices_bboxes = []
    #     for V in range(num_view):
    #         cor_y = torch.clamp(vertices_reproj[V, :, 1], min=0, max=h_np-1).long()
    #         cor_x = torch.clamp(vertices_reproj[V, :, 0], min=0, max=w_np-1).long()
    #         # cor_y_exp = torch.cat([cor_y, cor_y+1, cor_y-1, cor_y+1, cor_y-1], 0)
    #         # cor_x_exp = torch.cat([cor_x, cor_x+1, cor_x-1, cor_x-1, cor_x+1], 0)

            
    #         image_cur = images[V].index_put((cor_y, cor_x), torch.tensor([0., 0., 255.], device=device, dtype=torch.float32))
    #         dummy_img = torch.full((224, 224, 3), 255., device=device, dtype=torch.float32)
    #         x0 = torch.min(vertices_reproj[V, :, 0]).cpu().detach().numpy()
    #         y0 = torch.min(vertices_reproj[V, :, 1]).cpu().detach().numpy()
    #         x1 = torch.max(vertices_reproj[V, :, 0]).cpu().detach().numpy()
    #         y1 = torch.max(vertices_reproj[V, :, 1]).cpu().detach().numpy()
    #         # print(x0, y0, x1, y1)

    #         center = np.array([(x0+x1)/2, (y0+y1)/2])
    #         scale = np.array([1.1*max(x1-x0, y1-y0) / 200., 1.1*max(x1-x0, y1-y0) / 200.])
            
    #         crop_trans = torch.from_numpy(get_affine_transform(center, scale, 0, (224, 224))).to(device).float()
    #         vertices_bbox = torch.einsum('ij,kj->ki', crop_trans,
    #                                     vertices_reproj[V, :, :])
    #         image_bbox = dummy_img.index_put((vertices_bbox[:, 1].long(), vertices_bbox[:, 0].long()), torch.tensor([0., 0., 255.], device=device, dtype=torch.float32))
    #         center_tensor = torch.from_numpy(center).to(device).float()
    #         scale_tensor = torch.from_numpy(scale)[0].to(device).float()
    #         # print(vertices_bbox.shape)
    #         # print(vertices_bbox)
    #         # vertices_bbox[:, :2] = 2. * vertices_bbox[:, :2] / 224. - 1.
    #         bbox_info = torch.stack([center_tensor[0] - img_size[B][1] / 2., center_tensor[1] - img_size[B][0] / 2., scale_tensor]).unsqueeze(0)
    #         bbox_info[:, :2] = bbox_info[:, :2] / focal_length[B] * 2.8  # [-1, 1]
    #         bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length[B]) / (0.06 * focal_length[B])  # [-1, 1]
    #         # print(new_img.shape)
    #         # print(bbox_info.shape)
    #         vertices_bboxes.append(vertices_bbox.unsqueeze(0))
    #         bbox_infos.append(bbox_info)
    #         # print(delta_rot[B])
    #         # cv2.imwrite('/home/lab345/PycharmProjects/SPIN/img/rendered_img-{:.2f}-{:.2f}-{:.2f}-{:.1f}_full.png'.format(t[V][0].item(), t[V][1].item(), t[V][2].item(), delta_rot[B]), image_cur.cpu().detach().numpy())
    #         # cv2.imwrite('/home/lab345/PycharmProjects/SPIN/img/rendered_img-{:.2f}-{:.2f}-{:.2f}-{}.png'.format(t[V][0].item(), t[V][1].item(), t[V][2].item(),delta_rot[B]), image_bbox.cpu().detach().numpy())
    #     # print(vertices_bboxes)
    #     vertices_bboxes=torch.cat(vertices_bboxes, 1)
    #     bbox_infos=torch.cat(bbox_infos, 1)
    #     # print(new_imgs.shape)
    #     # print(bbox_infos.shape)

    #     vertices_bboxes_all.append(vertices_bboxes)
    #     bbox_infos_all.append(bbox_infos)
    
    # vertices_bboxes_all=torch.cat(vertices_bboxes_all, 0)
    # bbox_infos_all=torch.cat(bbox_infos_all, 0)
    # # print(vertices_bboxes_all)
    # # print(bbox_infos_all)
    # return vertices_bboxes_all, bbox_infos_all


if __name__ == '__main__':
    import torch

    images = torch.ones((4, 224, 224, 3), device=torch.device('cuda'), dtype=torch.float32, requires_grad=True) * 255.

    
    # cor_x = 10 *torch.randn(size=(10,), device=images.device, requires_grad=True).long()
    # cor_y = 10 *torch.randn(size=(10,), device=images.device, requires_grad=True).long()
    # img = images[V].index_put((cor_y, cor_x), torch.tensor([0., 0., 255.], device=images.device, dtype=torch.float32))
    # img.backward(torch.ones(images[V].shape, device=images.device, dtype=torch.float32), retain_graph=True)
    # print(cor_x.grad)

    qvalues = torch.zeros((5, 5), requires_grad=True)
    x = torch.LongTensor([1, 3])
    y = torch.LongTensor([0, 0]).requires_grad_()
    # qvalues[:, y] = torch.tensor([0., 0., 255.])
    new = qvalues.index_put((y, x), torch.tensor([255.], dtype=torch.float32))
    print(new.requires_grad) # True
    # qvalues_a = new ** 2

    new.backward(torch.ones(qvalues.shape), retain_graph=True)

    print(new)
    print(y.requires_grad)
    print(qvalues.grad)
        