import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=[224, 224], faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                       viewport_height=img_res[1],
                                       point_size=1.0)
        # print(img_res)
        self.focal_length = focal_length
        self.camera_center = [img_res[0] // 2, img_res[1] // 2]
        self.faces = faces

    def visualize_tb(self, pred_vertices, gt_vertices, camera_translation, images, base_color=(0.99, 0.83, 0.5, 1.0), amb_color=[0., 0., 0.], grid=True, blank_bg=False, side=None, save_path=None):
        pred_vertices = pred_vertices.cpu().numpy().copy()
        gt_vertices = gt_vertices.cpu().numpy().copy()
        camera_translation = camera_translation.cpu().numpy().copy()
        print('in render',images.shape)
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        if blank_bg:
            images_np = np.ones_like(images_np)
            # images_np = np.zeros_like(images_np)
        rend_imgs = []
        for i in range(pred_vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(pred_vertices[i], camera_translation[i], images_np[i], base_color=base_color, amb_color=amb_color, aroundy=side, save_path=save_path), (2,0,1))).float()
            gt_img = torch.from_numpy(np.transpose(self.__call__(gt_vertices[i], camera_translation[i], images_np[i], base_color=(0.1, 0.9, 0.1, 1.0)), (2,0,1))).float()
            if not grid:
                return rend_img
            print("griding")
            gt_img = images[i]
            rend_imgs.append(gt_img)
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, base_color=(0.99, 0.83, 0.5, 1.0), amb_color=[0., 0., 0.], aroundy=None, save_path=None):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=base_color,
            alphaCutoff=0.5)
            # baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] = camera_translation[0] * -1.
        if aroundy is not None:
            center = vertices.mean(axis=0)
            rot_vertices = np.dot((vertices - center), aroundy) + center
            vertices = rot_vertices

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

        scene = pyrender.Scene(ambient_light=amb_color)
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1., 1., 1.], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([-1, -1, -2])
        scene.add(light, pose=light_pose)

        
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        # print(valid_mask.shape)
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
