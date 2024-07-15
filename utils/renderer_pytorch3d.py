import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import numpy as np
import pytorch3d
import pytorch3d.renderer
import torch
from scipy.spatial.transform import Rotation
import cv2
from typing import List, Optional

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
        self.vert_per_mesh = 6890
        self.light_nodes = create_raymond_lights()



    def visualize_tb(self, pred_vertices, camera_translation, images, base_color=[(0.99, 0.83, 0.5, 1.0),], amb_color=[0., 0., 0.], grid=False, blank_bg=False, merge=True, side=None, gt_vertices=None, save_path=None):
        pred_vertices = pred_vertices.cpu().numpy()
        if gt_vertices is not None:
            gt_vertices = gt_vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        print('in render',images.shape)
        images = images.cpu()
        # images_np = np.transpose(images.numpy(), (0,2,3,1))
        images_np = images.numpy()
        if blank_bg:
            images_np = np.ones_like(images_np)
            # images_np = np.zeros_like(images_np)
        rend_imgs = []
        for i in range(pred_vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(pred_vertices[i], camera_translation[i], images_np[i], base_color=base_color, amb_color=amb_color, merge=merge, aroundy=side, save_path=save_path), (2,0,1))).float()
            # gt_img = torch.from_numpy(np.transpose(self.__call__(gt_vertices[i], camera_translation[i], images_np[i], color=color), (2,0,1))).float()
            if not grid:
                return rend_img
            
            gt_img = images[i]
            rend_imgs.append(gt_img)
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, base_color=[(0.99, 0.83, 0.5, 1.0),], amb_color=[0., 0., 0.], merge=False, aroundy=None, save_path=None):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=base_color[0],
            alphaCutoff=0.5)
            # baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.
        scene = pyrender.Scene(ambient_light=amb_color)
        if not merge:
            if aroundy is not None:
                center = vertices.mean(axis=0)
                rot_vertices = np.dot((vertices - center), aroundy) + center
                vertices = rot_vertices

            mesh = trimesh.Trimesh(vertices, self.faces)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
            # scene = pyrender.Scene(ambient_light=amb_color)
            scene.add(mesh, 'mesh')
        else:
            mesh_num = vertices.shape[0] // self.vert_per_mesh
            for m in range(mesh_num):
                verts = vertices[m*self.vert_per_mesh: (m+1)*self.vert_per_mesh]
                if aroundy is not None:
                    center = verts.mean(axis=0)
                    rot_verts = np.dot((verts - center), aroundy) + center
                    verts = rot_verts

                mesh = trimesh.Trimesh(verts, self.faces)
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(180), [1, 0, 0])
                material_single = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                alphaMode='OPAQUE',
                baseColorFactor=base_color[m][:3])
                mesh.apply_transform(rot)
                if save_path:
                    mesh.export(save_path[m])
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material_single, smooth=True)

                # scene = pyrender.Scene(ambient_light=amb_color)
                scene.add(mesh)
                # print(m)

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
    
    # def render_mesh(self, vertices, camera_translation, image=None, focal=None, base_color=[(1, 1, 1)], blank_bg=False, merge=False, side=None) :
    #     """
    #     Render meshes on input image
    #     Args:
    #         vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
    #         camera_translation (np.array): Array of shape (3,) with the camera translation.
    #         image (np.array): Array of shape (H, W, 3) containing the image crop with normalized pixel values.
    #     """

    #     height, width = image.shape[:2]

    #     scene = pyrender.Scene(bg_color=(0., 0., 0., 0.), ambient_light=(1.0, 1.0, 1.0))
    #     camera_translation = camera_translation.cpu().numpy()
    #     if image is not None:
    #         image_np = image[0].cpu().numpy()
    #     vertices = vertices.cpu().numpy()
    #     camera_translation = np.array(camera_translation) # also make a copy
    #     camera_translation_zero = np.zeros_like(camera_translation)
    #     camera_translation[0] *= -1.
    #     aroundy = side
    #     if aroundy is not None:
    #         center = vertices.mean(axis=0)
    #         rot_vertices = np.dot((vertices - center), aroundy) + center
    #         vertices = rot_vertices

    #     # Create mesh
    #     if len(vertices.shape) == 2:
    #         vertices = vertices[None]

    #     if merge:
    #         mesh_num = vertices.shape[1] // self.vert_per_mesh
    #         for m in range(mesh_num):
    #             vert = vertices[0, m*self.vert_per_mesh: (m+1)*self.vert_per_mesh]
    #             # vert = vertices[m]
    #             print(vert.shape)
    #             mesh = trimesh.Trimesh(vert, self.faces, process=False)
    #             rot = trimesh.transformations.rotation_matrix(
    #                 np.radians(180), [1, 0, 0])

    #             material_single = pyrender.MetallicRoughnessMaterial(
    #             metallicFactor=0.1,
    #             alphaMode='OPAQUE',
    #             baseColorFactor=base_color[m][:3])

    #             mesh.apply_transform(rot)
    #             mesh = pyrender.Mesh.from_trimesh(mesh, material=material_single, smooth=True)
    #             scene.add(mesh)
    #     else:
    #         pass
    #     # Create camera
    #     camera_pose = np.eye(4)
    #     camera_pose[:3, 3] = camera_translation
    #     camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
    #                                        cx=self.camera_center[0], cy=self.camera_center[1])
    #     scene.add(camera, pose=camera_pose)
    #     if blank_bg:
    #         image_np = np.ones_like(image_np)
    #     # Create light
    #     for node in self.light_nodes: scene.add_node(node)

    #     # Render
    #     color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.NONE)
    #     output_img = color[:, :, :3]
    #     return output_img
    #     color = color.astype(np.float32)
    #     valid_mask = (rend_depth > 0)[:,:,None]
    #     # print(valid_mask.shape)
    #     output_img = (color[:, :, :3] * valid_mask +
    #               (1 - valid_mask) * image_np)
    #     # # Composite
    #     # if image is None:
    #     #     output_img = color[:, :, :3]
    #     # else:
    #     #     valid_mask = (rend_depth > 0)[:, :, np.newaxis].astype(np.uint8)
    #     #     output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image) 
            

    #     return output_img

def dump_obj(verts, faces, save_path):
    with open(save_path, 'w') as fp:
        verts_np = verts.detach().cpu().numpy()
        print(verts_np.shape)
        print(faces.shape)
        for v in verts_np:
            fp.write('v %f %f %f\n' % (v[0],v[1],v[2]))         
        for f in faces + 1 :
            fp.write('f %d %d %d\n' % (f[0],f[1],f[2]))    


def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes
    
class Renderer_Pytorch3d:
    def __init__(self, focal_length=5000, img_res=[224, 224], faces=None, vert_per_mesh=6890):
        # print(img_res)
        self.focal_length = focal_length
        self.camera_center = [img_res[0] // 2, img_res[1] // 2]
        self.height = img_res[1]
        self.width = img_res[0]
        self.faces = faces.unsqueeze(0)
        self.vert_per_mesh = vert_per_mesh

    def render_mesh(self, vertices, translation, images, base_color=(0.99, 0.83, 0.5), grid=True, blank_bg=False, side=None, merge=False, device=None):
        ''' Render the mesh under camera coordinates
        vertices: (N_v, 3), vertices of mesh
        faces: (N_f, 3), faces of mesh
        translation: (3, ), translations of mesh or camera
        focal_length: float, focal length of camera
        height: int, height of image
        width: int, width of image
        device: "cpu"/"cuda:0", device of torch
        :return: the rgba rendered image
        '''
        if device is None:
            device = vertices.device

        bs = vertices.shape[0]

        # add the translation
        translation_zero = torch.zeros_like(translation).to(device)
        vertices = vertices + translation[:, None, :]

        # upside down the mesh
        # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
        rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
        rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
        # self.faces = self.faces.expand(bs, *self.faces.shape).to(device)
        self.faces = self.faces.to(device)
        images = images.cpu().numpy()

        vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)

        if side is not None:
            side = torch.from_numpy(side).to(torch.float32).to(device).unsqueeze(0)
            center = vertices.mean(axis=1)
            rot_vertices = torch.matmul((vertices - center), side) + center
            vertices = rot_vertices

        if not merge:
        # Initialize each vertex to be white in color.
            # verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
            verts_rgb = torch.FloatTensor(base_color[:3]).repeat(self.vert_per_mesh, 1).unsqueeze(0).to(device)
            textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
            mesh = pytorch3d.structures.Meshes(verts=vertices, faces=self.faces, textures=textures)
        else:
            mesh_list = []
            mesh_num = vertices.shape[1] // self.vert_per_mesh
            for m in range(mesh_num):
                vertices_single = vertices[:, m*self.vert_per_mesh: (m+1)*self.vert_per_mesh]
                # print(vertices_single.shape)
                # verts_rgb = torch.ones_like(vertices_single)  # (B, V, 3)
                verts_rgb = torch.FloatTensor(base_color[m][:3]).repeat(self.vert_per_mesh, 1).unsqueeze(0).to(device)
                # print(verts_rgb.shape)
                textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
                mesh_single = pytorch3d.structures.Meshes(verts=vertices_single, faces=self.faces, textures=textures)
                mesh_list.append(mesh_single)
            mesh = pytorch3d.structures.join_meshes_as_scene(mesh_list, include_textures=True)

        # Initialize a camera.
        cameras = pytorch3d.renderer.PerspectiveCameras(
            focal_length=((2 * self.focal_length / min(self.height, self.width), 2 * self.focal_length / min(self.height, self.width)),),
            device=device,
        )

        # Define the settings for rasterization and shading.
        raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=(self.height, self.width),   # (H, W)
            # image_size=height,   # (H, W)
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )

        # Define the material
        # ambient_color = (base_color[:3],) if not merge else ((1, 1, 1),)
        materials = pytorch3d.renderer.Materials(
            ambient_color=((1, 1, 1),),
            diffuse_color=((1, 1, 1),),
            specular_color=((1, 1, 1),),
            shininess=64,
            device=device
        )

        # Place a directional light in front of the object.
        lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

        # Create a phong renderer by composing a rasterizer and a shader.
        renderer = pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=pytorch3d.renderer.SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                materials=materials
            )
        )

        # Do rendering
        imgs = renderer(mesh).cpu().numpy() * 255.
        if blank_bg:
            input_img = np.ones_like(images) * 255.
        else:
            input_img = images

        color_batch = imgs

        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch)

        color = image_vis_batch
        valid_mask = valid_mask_batch
        
        # image_vis = input_img
        if not merge:
            alpha = 0.95
        else:
            alpha = 0.95
        # image_vis = ((1 - alpha) * input_img + alpha * input_img)
        image_vis = alpha * color[..., :3] * valid_mask + (
            1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

        # image_vis = image_vis.astype(np.uint8)
        # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        # res_path = os.path.join(opt.out_dir, basename)
        # cv2.imwrite(res_path, image_vis)
        return image_vis[0, ..., :3]
