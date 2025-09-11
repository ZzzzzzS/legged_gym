import torch
from typing import Tuple
import numpy as np
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils import torch_jit_utils, math

from isaacgym.torch_utils import quat_apply 
import torch
from abc import ABC, abstractmethod
import math


def CreateWireframeSphereLines(
    radius: float, color_list: list, pose: torch.Tensor, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """create wireframe sphere lines

    Args:
        radius (float): radius of the sphere
        color (list): color of the sphere [r,g,b] (0~1)
        pose (torch.Tensor): pose of the sphere [x,y,z,quat_x,quat_y,quat_z,quat_w]
        device (str): device to use

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: wireframe sphere lines, color of the sphere
    """
    pose = pose.to(device)
    radius45 = radius * 0.7071067811865475
    wire = torch.zeros(4, 6, device=device, requires_grad=False, dtype=torch.float32)
    wire[0, :] = torch.tensor([-radius, 0, 0, radius, 0, 0], device=device, requires_grad=False, dtype=torch.float32)
    wire[1, :] = torch.tensor([0, -radius, 0, 0, radius, 0], device=device, requires_grad=False, dtype=torch.float32)
    wire[2, :] = torch.tensor(
        [-radius45, -radius45, 0, radius45, radius45, 0], device=device, requires_grad=False, dtype=torch.float32
    )
    wire[3, :] = torch.tensor(
        [-radius45, radius45, 0, radius45, -radius45, 0], device=device, requires_grad=False, dtype=torch.float32
    )

    color = torch.zeros(4, 3, device=device, requires_grad=False, dtype=torch.float32)
    color[0, :] = torch.tensor(color_list, device=device, requires_grad=False, dtype=torch.float32)
    color[1, :] = torch.tensor(color_list, device=device, requires_grad=False, dtype=torch.float32)
    color[2, :] = torch.tensor(color_list, device=device, requires_grad=False, dtype=torch.float32)
    color[3, :] = torch.tensor(color_list, device=device, requires_grad=False, dtype=torch.float32)

    wire_local = wire.view(-1, 3)
    wire_pose = pose.repeat(wire_local.shape[0], 1)
    wire_global = torch_jit_utils.quat_apply(wire_pose[:, 3:7], wire_local)
    wire_global = wire_global + wire_pose[:, :3]
    wire_global = wire_global.view(-1, 6)

    return wire_global, color


def drw_dbg_viz(gym, viewer, pos: torch.Tensor, lines: torch.Tensor, color: torch.Tensor):
    """draw debug visualizer with given position and pattern configuration

    Args:
        gym (_type_): isaacgym handler
        viewer (_type_): isaacgym viewer
        pos (torch.Tensor): drawing position 3dim(only translation) or 7dim(translation+rotation)
        lines (torch.Tensor): lines to be drawn
        color (torch.Tensor): line color
    """
    pos = pos.view(-1, pos.shape[-1])
    pos_dim = pos.shape[1]
    pos2 = pos.repeat(1, 2 * lines.shape[0]).view(-1, pos_dim)
    local_vertices = lines.view(-1, 3)
    local_vertices = local_vertices.repeat(pos.shape[0], 1)
    if pos2.shape[1] == 3:  # no rotation
        global_vertices = local_vertices + pos2
    else:
        global_vertices = torch_jit_utils.quat_apply(pos2[:, 3:7], local_vertices) + pos2[:, :3]
    global_vertices = global_vertices.view(-1, 6)
    color = color.repeat(global_vertices.shape[0], 1)
    gym.add_lines(viewer, None, global_vertices.shape[0], global_vertices.cpu().numpy(), color.cpu().numpy())


class BatchLineGeometry(ABC):

    @abstractmethod
    def vertices(self)->torch.Tensor:
        """ Numpy array of Vec3 with shape (num_lines(), 2) """

    @abstractmethod
    def colors(self):
        """ Numpy array of Vec3 with length num_lines() """
        
    def num_lines(self):
        """ Returns the number of lines in this geometry """
        return self.vertices().shape[0]
    
    def draw(self, gym, viewer, quat:torch.Tensor, pos:torch.Tensor):
        
        
        vertices = self.vertices()
        num_lines = vertices.shape[0]
        num_envs = quat.shape[0]
        num_vertices = num_lines * 2
        
        vertices = vertices.unsqueeze(0)
        vertices = vertices.repeat(quat.shape[0], 1, 1, 1)
        vertices = vertices.view(-1, 3)
        
        quat = quat.unsqueeze(1).repeat(1, num_vertices, 1)
        pos = pos.unsqueeze(1).repeat(1, num_vertices, 1)
        quat = quat.view(-1,4)
        pos = pos.view(-1,3)
        vertices_world = quat_apply(quat, vertices) + pos
        
        vertices_world = vertices_world.view(-1, 2, 3)
        
        colors = self.colors()
        colors = colors.unsqueeze(0)
        colors = colors.repeat(num_envs, 1, 1)
        colors = colors.view(-1, 3)
        
        vertice_np = vertices_world.cpu().numpy()
        colors_np = colors.cpu().numpy()
        gym.add_lines(viewer,None, vertices_world.shape[0], vertice_np, colors_np)


class AxesGeometry(BatchLineGeometry):
    def __init__(self, device, scale=1.0):
        verts = torch.zeros((3, 2, 3), device=device, dtype=torch.float32)
        verts[0][0] = torch.tensor((0, 0, 0),device=device, dtype=torch.float32)
        verts[0][1] = torch.tensor((scale, 0, 0),device=device, dtype=torch.float32)
        verts[1][0] = torch.tensor((0, 0, 0),device=device, dtype=torch.float32)
        verts[1][1] = torch.tensor((0, scale, 0),device=device, dtype=torch.float32)
        verts[2][0] = torch.tensor((0, 0, 0),device=device, dtype=torch.float32)
        verts[2][1] = torch.tensor((0, 0, scale),device=device, dtype=torch.float32)

        colors = torch.zeros((3, 3), device=device, dtype=torch.float32)
        colors[0] = torch.tensor((1.0, 0.0, 0.0),device=device, dtype=torch.float32)  # Red for X-axis
        colors[1] = torch.tensor((0.0, 1.0, 0.0),device=device, dtype=torch.float32)
        colors[2] = torch.tensor((0.0, 0.0, 1.0),device=device, dtype=torch.float32)
        
        self._colors = colors
        self._verts = verts

    def vertices(self):
        return self._verts

    def colors(self):
        return self._colors


class WireframeBoxGeometry(BatchLineGeometry):
    def __init__(self,device, xdim=1, ydim=1, zdim=1,  color=None):
        
        num_lines = 12
        if color is None:
            color =  torch.tensor((1, 0, 0), device=device, dtype=torch.float32)
        else:
            color = torch.tensor(color, device=device, dtype=torch.float32)
            if color.shape != (3,):
                raise ValueError('Expected color to be a 3-element vector!')
            
        color = color.unsqueeze(0).repeat(num_lines, 1)

        

        x = 0.5 * xdim
        y = 0.5 * ydim
        z = 0.5 * zdim

        verts = torch.zeros((num_lines, 2, 3), device=device, dtype=torch.float32)
        # top face
        verts[0][0] = torch.tensor((x, y, z), device=device, dtype=torch.float32)
        verts[0][1] = torch.tensor((x, y, -z), device=device, dtype=torch.float32)
        verts[1][0] = torch.tensor((-x, y, z), device=device, dtype=torch.float32)
        verts[1][1] = torch.tensor((-x, y, -z), device=device, dtype=torch.float32)
        verts[2][0] = torch.tensor((x, y, z), device=device, dtype=torch.float32)
        verts[2][1] = torch.tensor((-x, y, z), device=device, dtype=torch.float32)
        verts[3][0] = torch.tensor((x, y, -z), device=device, dtype=torch.float32)
        verts[3][1] = torch.tensor((-x, y, -z), device=device, dtype=torch.float32)
        # bottom face
        verts[4][0] = torch.tensor((x, -y, z), device=device, dtype=torch.float32)
        verts[4][1] = torch.tensor((x, -y, -z), device=device, dtype=torch.float32)
        verts[5][0] = torch.tensor((-x, -y, z), device=device, dtype=torch.float32)
        verts[5][1] = torch.tensor((-x, -y, -z), device=device, dtype=torch.float32)
        verts[6][0] = torch.tensor((x, -y, z), device=device, dtype=torch.float32)
        verts[6][1] = torch.tensor((-x, -y, z), device=device, dtype=torch.float32)
        verts[7][0] = torch.tensor((x, -y, -z), device=device, dtype=torch.float32)
        verts[7][1] = torch.tensor((-x, -y, -z), device=device, dtype=torch.float32)
        # verticals
        verts[8][0] = torch.tensor((x, y, z), device=device, dtype=torch.float32)
        verts[8][1] = torch.tensor((x, -y, z), device=device, dtype=torch.float32)
        verts[9][0] = torch.tensor((x, y, -z), device=device, dtype=torch.float32)
        verts[9][1] = torch.tensor((x, -y, -z), device=device, dtype=torch.float32)
        verts[10][0] = torch.tensor((-x, y, z), device=device, dtype=torch.float32)
        verts[10][1] = torch.tensor((-x, -y, z), device=device, dtype=torch.float32)
        verts[11][0] = torch.tensor((-x, y, -z), device=device, dtype=torch.float32)
        verts[11][1] = torch.tensor((-x, -y, -z), device=device, dtype=torch.float32)
        
        self._colors = color
        self.verts = verts

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors



class WireframeBBoxGeometry(BatchLineGeometry):

    def __init__(self,device, bbox, color=None):
        if bbox.shape != (2, 3):
            raise ValueError('Expected bbox to be a matrix of 2 by 3!')

        num_lines = 12
        if color is None:
            color =  torch.tensor((1, 0, 0), device=device, dtype=torch.float32)     
        else:
            color = torch.tensor(color, device=device, dtype=torch.float32)
            if color.shape != (3,):
                raise ValueError('Expected color to be a 3-element vector!')

        color = color.unsqueeze(0).repeat(num_lines, 1)

        min_x, min_y, min_z = bbox[0]
        max_x, max_y, max_z = bbox[1]

        verts = torch.zeros((num_lines, 2, 3), device=device, dtype=torch.float32)
        # top face
        verts[0][0] = torch.tensor((max_x, max_y, max_z), device=device, dtype=torch.float32)
        verts[0][1] = torch.tensor((max_x, max_y, min_z), device=device, dtype=torch.float32)
        verts[1][0] = torch.tensor((min_x, max_y, max_z), device=device, dtype=torch.float32)
        verts[1][1] = torch.tensor((min_x, max_y, min_z), device=device, dtype=torch.float32)
        verts[2][0] = torch.tensor((max_x, max_y, max_z), device=device, dtype=torch.float32)
        verts[2][1] = torch.tensor((min_x, max_y, max_z), device=device, dtype=torch.float32)
        verts[3][0] = torch.tensor((max_x, max_y, min_z), device=device, dtype=torch.float32)
        verts[3][1] = torch.tensor((min_x, max_y, min_z), device=device, dtype=torch.float32)

        # bottom face
        verts[4][0] = torch.tensor((max_x, min_y, max_z), device=device, dtype=torch.float32)
        verts[4][1] = torch.tensor((max_x, min_y, min_z), device=device, dtype=torch.float32)
        verts[5][0] = torch.tensor((min_x, min_y, max_z), device=device, dtype=torch.float32)
        verts[5][1] = torch.tensor((min_x, min_y, min_z), device=device, dtype=torch.float32)
        verts[6][0] = torch.tensor((max_x, min_y, max_z), device=device, dtype=torch.float32)
        verts[6][1] = torch.tensor((min_x, min_y, max_z), device=device, dtype=torch.float32)
        verts[7][0] = torch.tensor((max_x, min_y, min_z), device=device, dtype=torch.float32)
        verts[7][1] = torch.tensor((min_x, min_y, min_z), device=device, dtype=torch.float32)

        # verticals
        verts[8][0] = torch.tensor((max_x, max_y, max_z), device=device, dtype=torch.float32)
        verts[8][1] = torch.tensor((max_x, min_y, max_z), device=device, dtype=torch.float32)
        verts[9][0] = torch.tensor((max_x, max_y, min_z), device=device, dtype=torch.float32)
        verts[9][1] = torch.tensor((max_x, min_y, min_z), device=device, dtype=torch.float32)
        verts[10][0] = torch.tensor((min_x, max_y, max_z), device=device, dtype=torch.float32)
        verts[10][1] = torch.tensor((min_x, min_y, max_z), device=device, dtype=torch.float32)
        verts[11][0] = torch.tensor((min_x, max_y, min_z), device=device, dtype=torch.float32)
        verts[11][1] = torch.tensor((min_x, min_y, min_z), device=device, dtype=torch.float32)

        self._colors = color
        self.verts = verts

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class WireframeSphereGeometry(BatchLineGeometry):

    def __init__(self,device, radius=1.0, num_lats=8, num_lons=8, color=None, color2=None):
        
        num_lines = 2 * num_lats * num_lons
        
        if color is None:
            color =  torch.tensor((1, 0, 0), device=device, dtype=torch.float32)
            #color = color.unsqueeze(0).repeat(num_lines, 1)
        else:
            color = torch.tensor(color, device=device, dtype=torch.float32)
            if color.shape != (3,):
                raise ValueError('Expected color to be a 3-element vector!')

        if color2 is None:
            color2 =  torch.tensor((1, 0, 0), device=device, dtype=torch.float32)
            #color2 = color2.unsqueeze(0).repeat(num_lines, 1) 
        else:
            color2 = torch.tensor(color2, device=device, dtype=torch.float32)
            if color2.shape != (3,):
                raise ValueError('Expected color2 to be a 3-element vector!')

           
        verts = torch.zeros((num_lines, 2, 3), device=device, dtype=torch.float32)
        colors = torch.zeros((num_lines, 3), device=device, dtype=torch.float32)
        
        idx = 0

        ustep = 2 * torch.pi / num_lats
        vstep = torch.pi / num_lons

        u = 0.0
        for i in range(num_lats):
            v = 0.0
            for j in range(num_lons):
                x1 = radius * math.sin(v) * math.sin(u)
                y1 = radius * math.cos(v)
                z1 = radius * math.sin(v) * math.cos(u)

                x2 = radius * math.sin(v + vstep) * math.sin(u)
                y2 = radius * math.cos(v + vstep)
                z2 = radius * math.sin(v + vstep) * math.cos(u)

                x3 = radius * math.sin(v + vstep) * math.sin(u + ustep)
                y3 = radius * math.cos(v + vstep)
                z3 = radius * math.sin(v + vstep) * math.cos(u + ustep)

                verts[idx][0] = torch.tensor((x1, y1, z1), device=device, dtype=torch.float32)
                verts[idx][1] = torch.tensor((x2, y2, z2), device=device, dtype=torch.float32)
                colors[idx] = color

                idx += 1

                verts[idx][0] = torch.tensor((x2, y2, z2), device=device, dtype=torch.float32)
                verts[idx][1] = torch.tensor((x3, y3, z3), device=device, dtype=torch.float32)
                colors[idx] = color2

                idx += 1

                v += vstep
            u += ustep

        self._colors = colors
        self.verts = verts

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors