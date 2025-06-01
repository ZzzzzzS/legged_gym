# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, quat_rotate_inverse, quat_rotate, get_euler_xyz,quat_conjugate,quat_from_euler_xyz,quat_mul
from typing import Tuple
import torch
from torch import Tensor


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def quat_rotate_yaw(quat, vec):
    ori_shape = vec.shape
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    vec_ = vec.view(-1, 3)
    vec2 = quat_rotate(quat_yaw, vec_)
    vec2 = vec2.view(ori_shape)
    return vec2


# @ torch.jit.script
def quat_rotate_yaw_inverse(quat, vec):
    ori_shape = vec.shape
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    vec_ = vec.view(-1, 3)
    vec2 = quat_rotate_inverse(quat_yaw, vec_)
    vec2 = vec2.view(ori_shape)
    return vec2


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

def so3_from_quat(quat):
    """
    Convert quaternion to so(3) rotation vector.
    :param quat: Tensor of shape (..., 4) representing quaternion (x, y, z, w).
    :return: Tensor of shape (..., 3) representing so(3) rotation vector.
    """
    quat = normalize(quat)  # Ensure quaternion is normalized
    if quat.shape[-1] != 4:
        raise ValueError("Input quaternion must have shape (..., 4)")
    quat = quat.view(-1, 4)
    w = quat[:, 3]
    v = quat[:, :3]
    theta = 2 * torch.acos(w)  # Angle of rotation
    sin_theta = torch.sin(theta / 2)
    so3_vector = torch.zeros_like(v)
    # Avoid division by zero
    mask = sin_theta > 1e-6
    so3_vector[mask] = (theta[mask].unsqueeze(-1) * v[mask]) / sin_theta[mask].unsqueeze(-1)
    so3_vector[~mask] = 0.0  # Set to zero where sin_theta is too small
    # Reshape to match the expected output shape
    so3_vector = so3_vector.view(quat.shape[:-1] + (3,))  # Reshape to (..., 3)
    return so3_vector

def so3_to_quat(so3_vector):
    """
    Convert so(3) rotation vector to quaternion.
    :param so3_vector: Tensor of shape (..., 3) representing so(3) rotation vector.
    :return: Tensor of shape (..., 4) representing quaternion (x, y, z, w).
    """
    if so3_vector.shape[-1] != 3:
        raise ValueError("Input so(3) vector must have shape (..., 3)")
    
    theta = torch.norm(so3_vector, dim=-1, keepdim=True)
    half_theta = theta / 2
    sin_half_theta = torch.sin(half_theta)
    
    w = torch.cos(half_theta)
    v = torch.zeros_like(so3_vector)
    mask = theta > 1e-6  # Avoid division by zero
    v[mask] = so3_vector[mask] * (sin_half_theta[mask] / theta[mask])  # Scale by sin(theta/2) / theta
    v[~mask] = 0.0  # Set to zero where theta is too small
    w[~mask] = 1.0  # Set w to 1 where theta is too small (identity quaternion)
    
    quat = torch.cat((v, w), dim=-1)  # Concatenate v and w to form quaternion
    return quat.view(so3_vector.shape[:-1] + (4,))  # Reshape to (..., 4)

def quat_slerp(quat1, quat2, t):
    """
    Spherical linear interpolation (slerp) between two quaternions.
    :param quat1: Tensor of shape (..., 4) representing the first quaternion.
    :param quat2: Tensor of shape (..., 4) representing the second quaternion.
    :param t: Tensor of shape (...) representing the interpolation factor (0 <= t <= 1).
    :return: Tensor of shape (..., 4) representing the interpolated quaternion.
    """
    t=t.unsqueeze(-1)  # Ensure t is a 1D tensor
    t=t.expand_as(quat1)  # Expand t to match the shape of quat1
    quat1 = normalize(quat1)
    quat2 = normalize(quat2)
    if quat1.shape[-1] != 4 or quat2.shape[-1] != 4:
        raise ValueError("Input quaternions must have shape (..., 4)")

    dot_product = torch.sum(quat1 * quat2, dim=-1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Ensure dot product is in [-1, 1]
    
    # rotate the second quaternion if the dot product is negative
    mask = dot_product < 0
    quat2 = torch.where(mask, -quat2, quat2)
    dot_product = torch.where(mask, -dot_product, dot_product)
    
    theta = torch.acos(dot_product)
    theta = theta.expand_as(quat1)  # Expand theta to match the shape of quat1
    sin_theta = torch.sin(theta)
    sin_theta.expand_as(quat1)  # Expand sin_theta to match the shape of quat1
    
    mask2 = sin_theta < 1e-6  # Handle cases where sin(theta) is too small
    mask2 = mask2.expand_as(quat1)
    
    quat_interpolated = torch.zeros_like(quat1)
    quat_interpolated[mask2] = quat1[mask2] + (quat2[mask2] - quat1[mask2]) * t[mask2]  # Linear interpolation
    quat_interpolated[~mask2] = (
        torch.sin((1 - t[~mask2]) * theta[~mask2]) / sin_theta[~mask2] * quat1[~mask2] +
        torch.sin(t[~mask2] * theta[~mask2]) / sin_theta[~mask2] * quat2[~mask2]
    )
    
    return normalize(quat_interpolated)

def quat_diff(quat_a, quat_b):
    quat_a = normalize(quat_a)
    quat_b = normalize(quat_b)
    dot = torch.sum(quat_a * quat_b, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)  # Ensure dot product is in [-1, 1]
    # If the dot product is negative, negate quat_b to ensure shortest path
    mask = dot < 0
    quat_b = torch.where(mask, -quat_b, quat_b)
    return quat_mul(quat_conjugate(quat_a), quat_b)
    