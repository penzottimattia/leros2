# Copyright 2025 Nicolas Gres
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any

from geometry_msgs.msg import PoseStamped

from leros2.components.common import ActionComponentConfig, ActionTopicComponent
from leros2.components.common.base import BaseComponentConfig
from leros2.components.common.rotation import RotationRepresentation


@ActionComponentConfig.register_subclass("pose_action")
@dataclass
class PoseActionComponentConfig(ActionComponentConfig):
    # name to identify the pose component
    name: str

    # ros2 frame id
    frame_id: str

    # representation of the orientation features (see ``RotationRepresentation``)
    rotation: RotationRepresentation = RotationRepresentation.QUATERNION

    # LeRobot 0.6.1 rollout only routes scalar features whose names end in
    # ``.pos``. Enable the same compatibility naming used by pose_state.
    lerobot_rollout_compat: bool = False


class PoseActionComponent(ActionTopicComponent[PoseActionComponentConfig, PoseStamped]):
    """Adapter for converting action features to a ROS 2 ``PoseStamped``.

    The position is read from ``<name>.pos.{x,y,z}``; the orientation keys
    depend on the configured :class:`RotationRepresentation` and are converted
    back into the quaternion the message carries.
    """

    def __init__(self, config: PoseActionComponentConfig):
        super().__init__(config, PoseStamped)

        self._rotation = config.rotation.encoding

    @property
    def features(self) -> dict[str, type | tuple[type, ...]]:
        name = self._config.name
        features = {
            f"{name}.pos.x": float,
            f"{name}.pos.y": float,
            f"{name}.pos.z": float,
            **self._rotation.features(name),
        }
        if not self._config.lerobot_rollout_compat:
            return features
        return {f"{key}.pos": value for key, value in features.items()}

    def to_message(self, action: dict[str, Any]) -> PoseStamped:
        name = self._config.name
        if self._config.lerobot_rollout_compat:
            action = {
                key.removesuffix(".pos"): value for key, value in action.items()
            }

        msg = PoseStamped()

        if self._node:
            msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._config.frame_id

        msg.pose.position.x = action[f"{name}.pos.x"]
        msg.pose.position.y = action[f"{name}.pos.y"]
        msg.pose.position.z = action[f"{name}.pos.z"]

        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = self._rotation.decode(name, action)

        return msg
