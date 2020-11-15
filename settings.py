# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import attr
import magnum as mn
import habitat_sim
import habitat_sim.agent
import habitat_sim.bindings as hsim

default_sim_settings = {
    # settings shared by example.py and benchmark.py
    "max_frames": 20,
    "width": 512,
    "height": 384,
    "default_agent": 0,
    "sensor_height": 1.5,
    "color_sensor": True,  # RGB sensor (default: ON)
    "semantic_sensor": False,  # semantic sensor (default: OFF)
    "depth_sensor": True,  # depth sensor (default: OFF)
    "seed": 1,
    "silent": True,  # do not print log info (default: OFF)
    # settings exclusive to example.py
    "save_png": True,  # save the pngs to disk (default: OFF)
    "print_semantic_scene": False,
    "print_semantic_mask_stats": False,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "scene": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    "test_scene_data_url": "http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip",
    "goal_position": [5.047, 0.199, 11.145],
    "goal_headings": [[0, -0.980785, 0, 0.195090], [0.0, 1.0, 0.0, 0.0]],
    "enable_physics": True,
    "num_objects": 10,
    "test_object_index": 0,
}

@attr.s(auto_attribs=True, slots=True)
class SpinSpec:
    spin_amount: float

@habitat_sim.registry.register_move_fn(body_action=True)
class SpinY(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: SpinSpec
    ):
        # Rotate about the +y (up) axis
        rotation_ax = habitat_sim.geo.UP
        scene_node.rotate_local(mn.Deg(actuation_spec.spin_amount), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()

@habitat_sim.registry.register_move_fn(body_action=True)
class SpinZ(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: SpinSpec
    ):
        # Rotate about the +y (up) axis
        rotation_ax = habitat_sim.geo.BACK
        scene_node.rotate_local(mn.Deg(actuation_spec.spin_amount), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()

@habitat_sim.registry.register_move_fn(body_action=True)
class SpinX(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: SpinSpec
    ):
        # Rotate about the +y (up) axis
        rotation_ax = habitat_sim.geo.RIGHT
        scene_node.rotate_local(mn.Deg(actuation_spec.spin_amount), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()

habitat_sim.registry.register_move_fn(
    SpinX, name="spin_x", body_action=True
)

habitat_sim.registry.register_move_fn(
    SpinY, name="spin_y", body_action=True
)

habitat_sim.registry.register_move_fn(
    SpinZ, name="spin_z", body_action=True
)


# build SimulatorConfiguration
def make_cfg(scene):
    default_sim_settings["scene"] = scene
    settings = default_sim_settings
    sim_cfg = hsim.SimulatorConfiguration()
    if "enable_physics" in settings.keys():
        sim_cfg.enable_physics = settings["enable_physics"]
    else:
        sim_cfg.enable_physics = False
    if "physics_config_file" in settings.keys():
        sim_cfg.physics_config_file = settings["physics_config_file"]
    print("sim_cfg.physics_config_file = " + sim_cfg.physics_config_file)
    sim_cfg.gpu_device_id = 0
    #print(settings["scene"])
    sim_cfg.scene.id = settings["scene"]

    # define default sensor parameters (see src/esp/Sensor/Sensor.h)
    sensors = {
        "color_sensor": {  # active if sim_settings["color_sensor"]
            "sensor_type": hsim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {  # active if sim_settings["depth_sensor"]
            "sensor_type": hsim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "semantic_sensor": {  # active if sim_settings["semantic_sensor"]
            "sensor_type": hsim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
    }

    # create sensor specifications
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = hsim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.gpu2gpu_transfer = False
            if not settings["silent"]:
                print("==== Initialized Sensor Spec: =====")
                print("Sensor uuid: ", sensor_spec.uuid)
                print("Sensor type: ", sensor_spec.sensor_type)
                print("Sensor position: ", sensor_spec.position)
                print("===================================")

            sensor_specs.append(sensor_spec)

    # create agent specifications
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_up": habitat_sim.agent.ActionSpec(
            "move_up", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_down": habitat_sim.agent.ActionSpec(
            "move_down", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "rotate_right": habitat_sim.ActionSpec(
            "spin_y", SpinSpec(36.0)
        ),
        "rotate_left": habitat_sim.ActionSpec(
            "spin_y", SpinSpec(-36.0)
        ),
        "rotate_up": habitat_sim.ActionSpec(
            "spin_z", SpinSpec(36.0)
        ),
        "rotate_down": habitat_sim.ActionSpec(
            "spin_z", SpinSpec(-36.0)
        ),
        "rotate_forward": habitat_sim.ActionSpec(
            "spin_x", SpinSpec(36.0)
        ),
        "rotate_backward": habitat_sim.ActionSpec(
            "spin_x", SpinSpec(-36.0)
        )
    }

    # override action space to no-op to test physics
    if sim_cfg.enable_physics:
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
            )
        }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
