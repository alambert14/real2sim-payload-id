import matplotlib.pyplot as plt
import numpy as np
import pydot
from IPython.display import SVG, display, Math
from pydrake.all import (AddMultibodyPlantSceneGraph, AngleAxis,
                         DiagramBuilder, FindResourceOrThrow, Integrator,
                         JacobianWrtVariable, LeafSystem, MeshcatVisualizer,
                         MultibodyPlant, MultibodyPositionToGeometryPose,
                         Parser, PiecewisePolynomial, PiecewisePose,
                         Quaternion, Rgba, RigidTransform, RotationMatrix,
                         SceneGraph, Simulator, StartMeshcat, TrajectorySource,
                         MeshcatAnimation, PassThrough, Demultiplexer,
                         MeshcatVisualizer, StartMeshcat, LogVectorOutput,
                         InverseDynamicsController, Adder, RollPitchYaw,
                         MakeVectorVariable, Variable, SpatialInertia_,
                         UnitInertia_, Expression, ToLatex, DecomposeLumpedParameters, RotationalInertia)

# from manipulation.meshcat_cpp_utils import StartMeshcat
# from manipulation.scenarios import MakeManipulationStation
from pydrake.autodiffutils import AutoDiffXd, ExtractValue
from pydrake.multibody import inverse_kinematics
from pydrake.multibody.tree import SpatialInertia, UnitInertia, FixedOffsetFrame
# from manipulation.meshcat_cpp_utils import AddMeshcatTriad
from pydrake.solvers import Solve
from pydrake.systems.framework import BasicVector

import symbolic_sysid_utils as sym_utils
import autodiff_sysid_utils as ad_utils


def create_planar_manipulation_station(q_traj, time_step=0, dim=2, with_object=False):
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    parser = Parser(plant)

    # Add arm
    arm = parser.AddModelFromFile(f"src/calibration/models/{dim}_dof_arm.urdf", "arm")

    if with_object:
        object = parser.AddModelFromFile(
            'src/calibration/models/payload.sdf', 'payload')
        joint = plant.WeldFrames(
            plant.GetFrameByName(f'link{dim}'),
            plant.GetFrameByName('payload', object),  # Mustard frame
            RigidTransform([0, 0, -1.5]),  # frame from parent to child TODO:(adjust for 1dof)
        )

    plant.Finalize()
    print(plant.num_positions(arm))
    meshcat = StartMeshcat()
    meshcat.Set2dRenderMode()
    animation = MeshcatAnimation()
    meshcat.SetAnimation(animation)

    # I need a PassThrough system so that I can export the input port.
    position = builder.AddSystem(PassThrough(dim * 2))
    builder.ExportOutput(position.get_output_port(),
                         "position_command")

    # Export the iiwa "state" outputs
    demux = builder.AddSystem(
        Demultiplexer(dim * 2, dim))
    builder.Connect(plant.get_state_output_port(arm), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "position_measured")
    builder.ExportOutput(demux.get_output_port(1), "velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(arm),
                         "arm_state_estimated")

    # Add controller plant
    controller_plant = MultibodyPlant(time_step)
    controller_parser = Parser(controller_plant)
    controller_arm = controller_parser.AddModelFromFile(f"src/calibration/models/{dim}_dof_arm.urdf")
    if with_object:
        object_ctrl = controller_parser.AddModelFromFile(
            'src/calibration/models/payload.sdf', 'payload')
        joint_ctrl = controller_plant.WeldFrames(
            controller_plant.GetFrameByName(f'link{dim}'),
            controller_plant.GetFrameByName('payload', object_ctrl),  # Mustard frame
            RigidTransform([0, 0, -1.5]),  # frame from parent to child
        )

    controller_plant.Finalize()

    # Add trajectory source
    if q_traj:
        traj = builder.AddSystem(TrajectorySource(q_traj, 1))  # Take the derivative once

    # Add Controller
    arm_controller = builder.AddSystem(
        InverseDynamicsController(controller_plant,
                                  kp=[100] * dim,
                                  kd=[10] * dim,
                                  ki=[1] * dim,
                                  has_reference_acceleration=False)
    )
    arm_controller.set_name("arm_controller")
    builder.Connect(plant.get_state_output_port(arm),
                    arm_controller.get_input_port_estimated_state())
    if q_traj:
        builder.Connect(traj.get_output_port(), position.get_input_port())


    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, dim))
    builder.Connect(arm_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values
    # if not connected).
    torque_passthrough = builder.AddSystem(PassThrough([0] * dim))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(),
                        "arm_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(arm))

    builder.Connect(position.get_output_port(),
                    arm_controller.get_input_port_desired_state())

    # Export commanded torques.
    builder.ExportOutput(adder.get_output_port(), "arm_torque_commanded")
    builder.ExportOutput(adder.get_output_port(), "arm_torque_measured")

    builder.ExportOutput(plant.get_generalized_contact_forces_output_port(arm),
                         "arm_torque_external")

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                         "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Attach trajectory
    # builder.Connect(q_source.get_output_port(),
    #                 iiwa_position.get_input_port())

    state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    torque_logger = LogVectorOutput(adder.get_output_port(), builder)
    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return plant, diagram, viz, meshcat, state_logger, torque_logger


def create_panda_manipulation_station(q_traj, time_step: object = 0, with_object: object = False, object_name: object = 'default') -> object:
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    parser = Parser(plant)

    # Add arm
    arm = parser.AddModelFromFile(f"src/calibration/models/panda_arm_no_collision.urdf", "arm")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"))

    if with_object:
        print('adding object')
        if object_name == 'default':
            object = parser.AddModelFromFile(
                'src/calibration/models/payload.sdf', 'payload')
            joint = plant.WeldFrames(
                plant.GetFrameByName(f'panda_link7'),
                plant.GetFrameByName('payload', object),  # is this the same as GetBodyByName?
                RigidTransform(RollPitchYaw(0, -np.pi / 2, 0), [0.0, 0.0, 0.2])
            )
        elif object_name == '009_gelatin_box':
            print('adding gelatin')
            object = parser.AddModelFromFile(
                'src/calibration/models/010_potted_meat_can.sdf', 'payload')

            print('Panda 7 Frame: ', plant.GetFrameByName('panda_link7').body())
            print('Object frame: ', plant.GetFrameByName('base_link_meat', object).body())

            joint = plant.WeldFrames(
                plant.GetFrameByName(f'panda_link7'),
                plant.GetFrameByName('base_link_meat', object),  # Mustard frame
                RigidTransform(RollPitchYaw(0, 0, 0), [0.0, 0.0, 0.15])
            )
    plant.Finalize()
    print(plant.num_positions(arm))

    meshcat = StartMeshcat()
    # meshcat.Set2dRenderMode()
    animation = MeshcatAnimation()
    meshcat.SetAnimation(animation)

    # I need a PassThrough system so that I can export the input port.
    position = builder.AddSystem(PassThrough(14))
    builder.ExportOutput(position.get_output_port(),
                         "position_command")

    # Export the iiwa "state" outputs
    demux = builder.AddSystem(
        Demultiplexer(14, 7))
    print(plant.get_state_output_port(arm).size())
    builder.Connect(plant.get_state_output_port(arm), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "position_measured")  # double check this
    builder.ExportOutput(demux.get_output_port(1), "velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(arm),
                         "arm_state_estimated")

    # Add controller plant
    controller_plant = MultibodyPlant(time_step)
    controller_parser = Parser(controller_plant)
    controller_arm = controller_parser.AddModelFromFile(f"src/calibration/models/panda_arm_no_collision.urdf")
    controller_plant.WeldFrames(controller_plant.world_frame(), controller_plant.GetFrameByName("panda_link0"))

    if with_object:
        if object_name == 'default':
            object_ctrl = controller_parser.AddModelFromFile(
                'src/calibration/models/payload.sdf', 'payload')
            joint_ctrl = controller_plant.WeldFrames(
                controller_plant.GetFrameByName(f'panda_link7'),
                controller_plant.GetFrameByName('payload', object_ctrl),  # is this the same as GetBodyByName?
                RigidTransform(RollPitchYaw(0, -np.pi / 2, 0), [0.0, 0.0, 0.2])
            )
        elif object_name == '009_gelatin_box':
            object_ctrl = controller_parser.AddModelFromFile(
                'src/calibration/models/010_potted_meat_can.sdf', 'payload')  # 009_gelatin_box.sdf
            joint_ctrl = controller_plant.WeldFrames(
                controller_plant.GetFrameByName(f'panda_link7'),
                controller_plant.GetFrameByName('base_link_meat', object_ctrl),  # Mustard frame
                RigidTransform(RollPitchYaw(0, 0, 0), [0.0, 0.0, 0.15])
            )

    controller_plant.Finalize()

    # Add trajectory source
    if q_traj:
        traj = builder.AddSystem(TrajectorySource(q_traj, 1))  # Take the derivative once

    # Add Controller
    arm_controller = builder.AddSystem(
        InverseDynamicsController(controller_plant,
                                  kp=[100] * 7,
                                  kd=[20] * 7,
                                  ki=[1] * 7,
                                  has_reference_acceleration=False)
    )
    arm_controller.set_name("arm_controller")
    builder.Connect(plant.get_state_output_port(arm),
                    arm_controller.get_input_port_estimated_state())
    if q_traj:
        builder.Connect(traj.get_output_port(), position.get_input_port())


    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, 7))
    builder.Connect(arm_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values
    # if not connected).
    torque_passthrough = builder.AddSystem(PassThrough([0] * 7))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(),
                        "arm_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(arm))

    builder.Connect(position.get_output_port(),
                    arm_controller.get_input_port_desired_state())

    # Export commanded torques.
    builder.ExportOutput(adder.get_output_port(), "arm_torque_commanded")
    builder.ExportOutput(adder.get_output_port(), "arm_torque_measured")

    builder.ExportOutput(plant.get_generalized_contact_forces_output_port(arm),
                         "arm_torque_external")

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                         "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    torque_logger = LogVectorOutput(adder.get_output_port(), builder)
    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return plant, diagram, viz, meshcat, state_logger, torque_logger


def purturb_plant(plant, std=None):
    """
    Add an object to a plant's links to see if we can identify
    :param plant:
    :param std: If none, return the same plant
    :return: Returns the same plant if std is None, or a new plant with added noise specified by std
    """
    context = plant.CreateDefaultContext()

    if std is None:
        return plant

    for body_idx in range(plant.num_bodies()):
        body = plant.get_body(body_idx)
        mass = body.get_mass(context)
        com = body.CalcCenterOfMassInBodyFrame(context)
        spa = body.CalcSpatialInertiaInBodyFrame()



    # TODO: Finish this

def add_point_to_plant(plant, context):
    """
    Add an object to the end of the pendulum to see if we can identify
    Mutates the context
    :param plant:
    :return:
    """
    link2 = plant.GetBodyByName("link2")
    spa = link2.CalcSpatialInertiaInBodyFrame(context)
    mass = link2.get_mass(context)
    com = link2.CalcCenterOfMassInBodyFrame(context)
    rot_inertia = spa.CalcRotationalInertia()
    point_inertia = RotationalInertia(1, [0, 0, 1])

    new_mass = mass + 1
    new_inertia = rot_inertia + point_inertia
    new_com = (mass * com + 1 * np.array([0, 0, 1])) / new_mass
    new_unit_inertia = UnitInertia()
    new_unit_inertia.SetFromRotationalInertia(new_inertia, new_mass)
    new_spa = SpatialInertia(new_mass, new_com, new_unit_inertia)
    link2.SetSpatialInertiaInBodyFrame(context, new_spa)

    return plant, context  # I don't think this is actually propagated through


## Modified from pangtao/pick-and-place-benchmarking-framework SimpleTrajectory
class PickAndPlaceTrajectorySource(LeafSystem):

    def __init__(self, plant: MultibodyPlant,
                 X_L7_start: RigidTransform, X_L7_end: RigidTransform, clearance: float = 0.3):
        super().__init__()
        self.plant = plant
        self.init_guess_start = np.array([0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785])
        self.init_guess_end = np.array([1.57, -1.285, 0, -2.356, 0.0, 1.571, 0.785])
        # AddMeshcatTriad(meshcat, "start",
        #                 length=0.15, radius=0.006, X_PT=X_L7_start)
        # AddMeshcatTriad(meshcat, "end",
        #                 length=0.15, radius=0.006, X_PT=X_L7_end)

        hover_start = RigidTransform([0, 0, 0.1])
        hover_end = RigidTransform([0, 0, -0.1])

        X_L7_start_hover = RigidTransform(X_L7_start.rotation(), X_L7_start.translation() + [0., 0., 0.1])
        X_L7_end_hover = RigidTransform(X_L7_end.rotation(), X_L7_end.translation() + [0., 0., 0.1])
        self.start_q = self.inverse_kinematics(X_L7_start, start=True)  # [:-1]
        self.start_hover_q = self.inverse_kinematics(X_L7_start_hover, start=True)
        self.end_hover_q = self.inverse_kinematics(X_L7_end_hover, start=False)
        self.end_q = self.inverse_kinematics(X_L7_end, start=False)  # [:-1]
        self.q_traj = self.calc_q_traj()
        print(self.q_traj.value(3))

        self.x_output_port = self.DeclareVectorOutputPort(
            'traj_x', BasicVector(self.q_traj.rows() * 2), self.calc_x)
        self.t_start = 0

    def inverse_kinematics(self, X_L7: RigidTransform, start=True):
        """
        Given a pose in the world, calculate a reasonable joint configuration for the KUKA iiwa arm that would place
        the end of link 7 in that position.
        :return: Joint configuration for the iiwa
        """
        ik = inverse_kinematics.InverseKinematics(self.plant)
        q_variables = ik.q()

        position_tolerance = 0.01
        frame_L7 = self.plant.GetFrameByName('panda_link7')
        # Position constraint
        p_L7_ref = X_L7.translation()
        ik.AddPositionConstraint(
            frameB=frame_L7, p_BQ=np.zeros(3),
            frameA=self.plant.world_frame(),
            p_AQ_lower=p_L7_ref - position_tolerance,
            p_AQ_upper=p_L7_ref + position_tolerance)

        # Orientation constraint
        R_WL7_ref = X_L7.rotation()  # RotationMatrix(R_WE_traj.value(t))
        ik.AddOrientationConstraint(
            frameAbar=self.plant.world_frame(),
            R_AbarA=R_WL7_ref,
            frameBbar=frame_L7,
            R_BbarB=RotationMatrix(),
            theta_bound=0.01)

        prog = ik.prog()
        # use the robot posture at the previous knot point as
        # an initial guess.
        if start:
            init_guess = self.init_guess_start
        else:
            init_guess = self.init_guess_end
        prog.SetInitialGuess(q_variables, init_guess)
        print(prog)
        result = Solve(prog)
        assert result.is_success()
        return result.GetSolution(q_variables)

    def calc_x(self, context, output):
        t = context.get_time() - self.t_start
        q = self.q_traj.value(t).ravel()
        v = self.q_traj.derivative(1).value(t).ravel()
        output.SetFromVector(np.hstack([q, v]))

    def set_t_start(self, t_start_new: float):
        self.t_start = t_start_new

    def calc_q_traj(self) -> PiecewisePolynomial:
        """
        Generate a joint configuration trajectory from a beginning and end configuration
        :return: PiecewisePolynomial
        """
        return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [0, 1, 9, 10], np.vstack([self.start_q, self.start_hover_q, self.end_hover_q, self.end_q]).T,
            np.zeros(7), np.zeros(7))