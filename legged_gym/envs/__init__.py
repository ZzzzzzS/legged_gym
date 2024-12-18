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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .cartpole2.cartpole2 import Cartpole2Task
from .cartpole2.cartpole2_config import Cartpole2Config, Cartpole2ConfigPPO
from .pf2.pf2_env import Pf2Env
from .pf2.pf2_config import Pf2Cfg, Pf2CfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register("anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO())
task_registry.register("anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO())
task_registry.register("anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO())
task_registry.register("a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
task_registry.register("cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO())
task_registry.register("cartpole2", Cartpole2Task, Cartpole2Config(), Cartpole2ConfigPPO())
task_registry.register("pf2", Pf2Env, Pf2Cfg(), Pf2CfgPPO())

robot_type = os.getenv("ROBOT_TYPE")
print(robot_type, "in env __init__")
# Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
if not robot_type:
    print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>' if using limx environment.")
    robot_type = "PF_TRON1A"

if robot_type.startswith("PF"):
    from .limx_tron1.PF.pointfoot import PointFoot
    from legged_gym.envs.limx_tron1.mixed_terrain.pointfoot_rough_config import PointFootRoughCfg, PointFootRoughCfgPPO
    from legged_gym.envs.limx_tron1.flat.PF.pointfoot_flat_config import PointFootFlatCfg, PointFootFlatCfgPPO

    task_registry.register("pointfoot_rough", PointFoot, PointFootRoughCfg(), PointFootRoughCfgPPO())
    task_registry.register("pointfoot_flat", PointFoot, PointFootFlatCfg(), PointFootFlatCfgPPO())
elif robot_type.startswith("WF"):
    from .limx_tron1.WF.pointfoot import PointFoot
    from legged_gym.envs.limx_tron1.mixed_terrain.pointfoot_rough_config import PointFootRoughCfg, PointFootRoughCfgPPO
    from legged_gym.envs.limx_tron1.flat.WF.pointfoot_flat_config import PointFootFlatCfg, PointFootFlatCfgPPO

    task_registry.register("pointfoot_flat", PointFoot, PointFootFlatCfg(), PointFootFlatCfgPPO())
elif robot_type.startswith("SF"):
    from .limx_tron1.SF.pointfoot import PointFoot
    from legged_gym.envs.limx_tron1.mixed_terrain.pointfoot_rough_config import PointFootRoughCfg, PointFootRoughCfgPPO
    from legged_gym.envs.limx_tron1.flat.SF.pointfoot_flat_config import PointFootFlatCfg, PointFootFlatCfgPPO

    task_registry.register("pointfoot_flat", PointFoot, PointFootFlatCfg(), PointFootFlatCfgPPO())
