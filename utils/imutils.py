"""
This file contains functions that are used to perform data augmentation.
"""
import torch
import numpy as np
# import scipy.misc
import cv2
from skimage.transform import resize, rotate

import constants


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.
    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.
    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)
    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.
    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian
    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt



def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False,
                         pixel_std=200.0):
    """Get the affine transform matrix, given the center/scale/rot/output_size.
    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    if scale.shape != (2, ):
        crop_ratio = output_size[1] / float(output_size[0])
        scale = np.array([scale/crop_ratio, scale])
    assert len(output_size) == 2
    assert len(shift) == 2

    scale_tmp = scale * pixel_std

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.
    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform
    Returns:
        np.ndarray: Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

    return new_pt



def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0, get_crop=False):
    """Transform pixel location to different reference."""
    if get_crop:
        crop_t = get_transform(center, scale, res, rot=0)
        return crop_t
    else:
        t = get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int)+1

def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,
                             res[1]+1], center, scale, res, invert=1))-1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        # new_img = scipy.misc.imrotate(new_img, rot)
        new_img = rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    # new_img = scipy.misc.imresize(new_img, res)
    new_img = resize(new_img, res)
    return new_img

def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,res[1]+1], center, scale, res, invert=1))-1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    # img = scipy.misc.imresize(img, crop_shape, interp='nearest')
    img = resize(img, crop_shape, order=0)
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    return new_img

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def flip_kp(kp, img_width=None):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = constants.J49_FLIP_PERM
    elif len(kp) == 45:
        flipped_parts = constants.J45_FLIP_PERM
    kp = kp[flipped_parts]
    if img_width is None:
        kp[:,0] = - kp[:,0]
    else:
        kp[:, 0] = img_width - 1 - kp[:, 0]
    return kp

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose

def bbox_from_keypoint(keypoints, rescale=1.2):
        """
        Get center and scale of bounding box from gt keypoints.
        The expected format is [24,3].
        """
        # print(keypoints)
        keypoints_valid = keypoints[np.where(keypoints[:, 2]>0)]
        if len(np.where(keypoints[:, 2]>0)[0]) == 0:
            print(keypoints)
        # print(np.where(keypoints[:, 2]>1), keypoints_valid)

        bbox = [min(keypoints_valid[:,0]), min(keypoints_valid[:,1]),
                        max(keypoints_valid[:,0]), max(keypoints_valid[:,1])]
        
        # center
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        center = np.array([center_x, center_y])

        # scale
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_size = max(bbox_w * constants.CROP_ASPECT_RATIO, bbox_h)
        scale = bbox_size / 200.0

        
        # adjust bounding box tightness
        scale *= rescale
        # print(center, scale)
        return center, scale

def bbox_conv_bboxXYHW_to_bbox_centerscale(bbox_XYWH, rescale=1.2, imageHeight= None):
    """
    Bbox conversion from bbox_xywh to (bbox_center, bbox_scale)
        # bbox_center: center of the bbox in original cooridnate
        # bbox_scale: scaling factor applied to the original image, before applying 224x224 cropping 

        # In the code: 200, and rescale ==1.2 are some magic numbers used for dataset generation in SPIN code
    """

    center = bbox_XYWH[:2] + 0.5 * bbox_XYWH[2:]
    bbox_size = max(bbox_XYWH[2:])
    # adjust bounding box tightness
    scale = bbox_size / 200.0           #2
    scale *= rescale
    return center, scale#, bbox_XYWH

def conv_bboxinfo_centerscale_to_bboxXYXY(center, scale, ratio):
    """
    from (center, scale) ->  (topleft, bottom right)  or (minX,minY,maxX,maxY)
    """

    # # hmr_res = (224,224)
    
    # """Crop image according to the supplied bounding box."""
    # # Upper left point
    # ul = np.array(transform([1, 1], center, scale, hmr_res, invert=1))-1
    # # Bottom right point
    # br = np.array(transform([hmr_res[0]+1,
    #                          hmr_res[1]+1], center, scale, hmr_res, invert=1))-1

    # return np.concatenate( (ul, br))
    h_tmp = scale * 200
    w_tmp = h_tmp / ratio
    p1 = np.array((center[0] - w_tmp/2., center[1] - h_tmp/2.))
    p2 = np.array((center[0] + w_tmp/2., center[1] + h_tmp/2.))
    return np.concatenate( (p1, p2))

def conv_bboxinfo_bboxXYHW_to_centerscale(bbox_xyhw, ratio, bLooseBox = False):
    """
    from (bbox_xyhw) -> (center, scale)
    Args:
        bbox_xyhw: [minX,minY,W,H]
        bLooseBox: if true, draw less tight box with sufficient margin (SPIN's default setting)
    Output:
        center: bbox center
        scale: scaling images before cropping. reference size is 200 pix (why??). >1.0 means size up, <1.0 means size down. See get_transform()
                h = 200 * scale
                t = np.zeros((3, 3))
                t[0, 0] = float(res[1]) / h
                t[1, 1] = float(res[0]) / h
                t[0, 2] = res[1] * (-float(center[0]) / h + .5)
                t[1, 2] = res[0] * (-float(center[1]) / h + .5)
                t[2, 2] = 1
    """

    center = [bbox_xyhw[0] + bbox_xyhw[2]/2, bbox_xyhw[1] + bbox_xyhw[3]/2]

    if bLooseBox:
        scaleFactor =1.2
        scale = scaleFactor*max(bbox_xyhw[2], bbox_xyhw[3])/200       #This is the one used in SPIN's pre-processing. See preprocessdb/coco.py
    else:
        scale = max(bbox_xyhw[2]*ratio, bbox_xyhw[3])/200   

    return center, scale


def multilvel_bbox_crop_gen(rawImg, fullsize_center, fullsize_scale, headVert, bbox_type):
    """
    Generate bbox from smallest size(face) to full size
    args:
        fullsize_center, fullsize_scale: bbox given by original annotation  (full body or maximum size)
        smpl_vert: 
    """
    if bbox_type == 'rect':
        ratio = constants.CROP_ASPECT_RATIO
    elif bbox_type == 'square':
        ratio = 1.
    bbox_list =[]
    
    bbox_xyxy_full = conv_bboxinfo_centerscale_to_bboxXYXY(fullsize_center, fullsize_scale, ratio=ratio)
    # cv2.rectangle(rawImg, (int(bbox_xyxy_full[0]), int(bbox_xyxy_full[1])), (int(bbox_xyxy_full[2]), int(bbox_xyxy_full[3])), thickness=8, color=(0, 255, 0))
    
    # Get face bbox (min size)
    
    minPt = [min(headVert[:,0]), min(headVert[:,1])] 
    maxPt = [max(headVert[:,0]), max(headVert[:,1])] 
    bbox_xyxy_small = [minPt[0],minPt[1], maxPt[0], maxPt[1]]

    #Interpolation
    minPt_d =  bbox_xyxy_full[:2] - bbox_xyxy_small[:2]
    maxPt_d =  bbox_xyxy_full[2:] - bbox_xyxy_small[2:]
    for i in range(8):
        cur_minPt = bbox_xyxy_small[:2] + minPt_d * i/7.0
        cur_maxPt = bbox_xyxy_small[2:] + maxPt_d * i/7.0
       
        bbox_xyhw = [cur_minPt[0],cur_minPt[1], cur_maxPt[0]-cur_minPt[0], cur_maxPt[1]-cur_minPt[1] ]
        cur_center,cur_scale = conv_bboxinfo_bboxXYHW_to_centerscale(bbox_xyhw, ratio=ratio)
        #Compute face to cur bbox ratio   cur_scale / face_scale
        if i==0:
            ratio_bbox_over_face = 1.0
        else:
            ratio_bbox_over_face = cur_scale/ bbox_list[0]['scale']
        bbox_list.append({"scale":cur_scale, "center": cur_center, "ratio_bbox_over_face": ratio_bbox_over_face})
    # cv2.rectangle(rawImg, (int(cur_minPt[0]), int(cur_minPt[1])), (int(cur_maxPt[0]), int(cur_maxPt[1])), thickness=2*i, color=(0, 255, 0))
    # cv2.imshow('head', rawImg)
    # cv2.waitKey(3000)
    return bbox_list