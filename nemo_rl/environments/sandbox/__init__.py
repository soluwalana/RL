# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sandboxed NeMo-Gym support for NeMo-RL.

The broker, wire contract, egress policy and job-host orchestration live in the
``sandboxed_gym`` package. Only :mod:`~nemo_rl.environments.sandbox.nemo_gym_actor`,
the NeMo-RL adapter over it, remains here.
"""
