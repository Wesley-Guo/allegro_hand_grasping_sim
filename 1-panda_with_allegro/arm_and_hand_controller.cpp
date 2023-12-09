// This controller controls both franka arm and allegro hand by running seperate controllers.  
#include "Sai2Model.h"
#include "Sai2Primitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"

#include <iostream>
#include <string>
#include <math.h>



// handle ctrl-c nicely
#include <signal.h>
bool runloop = true;
void sighandler(int sig)
{ runloop = false; }


using namespace std;
using namespace Eigen;

const std::string arm_robot_file = "./resources/panda_arm.urdf";
const std::string hand_robot_file = "./resources/allegro_hand.urdf";

// redis keys:
// - read:

// simulation keys for full arm + hand
const std::string JOINT_ANGLES_SIM_KEY = "sai2::panda_robot_with_allegro::sensors::q";
const std::string JOINT_VELOCITIES_SIM_KEY = "sai2::panda_robot_with_allegro::sensors::dq";

// real sensors for allegro 
const std::string ALLEGRO_JOINT_ANGLES_KEY = "allegroHand::sensors::joint_positions";
const std::string ALLEGRO_JOINT_VELOCITIES_KEY = "allegroHand::sensors::joint_velocities";

// real sensors for franka
const std::string FRANKA_JOINT_ANGLES_KEY  = "sai2::FrankaPanda::Bonnie::sensors::q";
const std::string FRANKA_JOINT_VELOCITIES_KEY = "sai2::FrankaPanda::Bonnie::sensors::dq";
const std::string FRANKA_MASSMATRIX_KEY = "sai2::FrankaPanda::Bonnie::sensors::model::massmatrix";
const std::string FRANKA_CORIOLIS_KEY = "sai2::FrankaPanda::Bonnie::sensors::model::coriolis";


std::string FINGERTIP_POSITION_KEY = "allegroHand::controller::finger_positions_commanded";

const std::string VR_RIGHT_CONTROLLER_ROTATION_KEY = "vr::right::rotation";
const std::string VR_RIGHT_CONTROLLER_POSITION_KEY = "vr::right::position";
const std::string VR_RIGHT_CONTROLLER_GRIP_KEY = "vr::right::grip_button";
const std::string VR_RIGHT_CONTROLLER_TRIGGER_KEY = "vr::right::trigger_button";

// values this key can take are "arm" or "hand"
const string CONTROL_MODE_KEY = "sai2::panda_robot_with_allegro::control_mode";

// - write

// simulation keys for arm + hand
const std::string ARM_TORQUES_COMMANDED_SIM_KEY = "sai2::panda_robot::actuators::tau";
const std::string HAND_TORQUES_COMMANDED_SIM_KEY = "sai2::allegro::actuators::tau";

// real torque commands for franka + allegro
const std::string FRANKA_COMMAND_TORQUES_KEY = "sai2::FrankaPanda::Bonnie::actuators::fgc";
const std::string ALLEGRO_COMMAND_TORQUES_KEY = "allegroHand::controller::joint_torques_commanded";

// real allegro palm-orinetation for gravity compensation
const string ALLEGRO_PALM_ORIENTATION = "allegroHand::controller::palm_orientation";

const std::string CONTROLLER_RUNNING = "sai2::panda_robot::controller_running";

// switch for simulation mode 
const bool flag_simulation = true;
unsigned long long controller_counter = 0;


// gain values for allegro
const double HAND_JOINT_KP = 125.0;
const double HAND_JOINT_KV = 45.0;

double HAND_V_MAX, HAND_TASK_POSITION_GAIN, HAND_TASK_VELOCITY_GAIN, HAND_POSTURE_POSITION_GAIN, HAND_POSTURE_VELOCITY_GAIN;


// control modes
enum class ControlMode {ARM , HAND};

int main() {

	// start redis client
	auto redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// initialize the control mode
	ControlMode control_mode = ControlMode::ARM;

	// Gain values for allegro hand
	if (flag_simulation) { 
		HAND_V_MAX = 1.0;
		HAND_TASK_POSITION_GAIN = 125.0;
		HAND_TASK_VELOCITY_GAIN = 2.25;

		HAND_POSTURE_POSITION_GAIN = 0.0025;
		HAND_POSTURE_VELOCITY_GAIN = 0.0015;
	} else {
		HAND_V_MAX = 1.5;
		HAND_TASK_POSITION_GAIN = 175.0;
		HAND_TASK_VELOCITY_GAIN = 2.0;

		HAND_POSTURE_POSITION_GAIN = 0.05;
		HAND_POSTURE_VELOCITY_GAIN = 0.025;
	}

	// VR to robot workspace transform
	const double device_z_rot = 180;
	Matrix3d R_vr_to_base;  // default vr: x right, y up, z in
	R_vr_to_base = AngleAxisd(M_PI / 2, Vector3d(1, 0, 0)).toRotationMatrix();  // x right, y left, z up
	Matrix3d R_base_to_user;
	R_base_to_user = AngleAxisd(device_z_rot * M_PI / 180, Vector3d(0, 0, 1)).toRotationMatrix();
	Matrix3d R_vr_to_user = R_base_to_user * R_vr_to_base;

    // Initialize franka robot model
	const string arm_link_name = "link7";
	const Vector3d arm_tcp_pos_in_link = Vector3d(0, 0, 0.1);

	auto arm_robot = new Sai2Model::Sai2Model(arm_robot_file, false);
	const int arm_dof = arm_robot->dof();

	// Read robot states and update the robot model
	VectorXd q = VectorXd::Zero(arm_dof + 16);
	VectorXd dq = VectorXd::Zero(arm_dof + 16);

	if (flag_simulation){
		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
		dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);
	}

	if (flag_simulation) {
		arm_robot->_q = q.head(arm_dof);
		arm_robot->_dq = dq.head(arm_dof);
	} else {
		arm_robot->_q = redis_client.getEigenMatrixJSON(FRANKA_JOINT_ANGLES_KEY);
		arm_robot->_dq = redis_client.getEigenMatrixJSON(FRANKA_JOINT_VELOCITIES_KEY);
	}
	
	VectorXd last_arm_q = arm_robot->_q;
	VectorXd initial_arm_q = arm_robot->_q;
	VectorXd arm_coriolis = VectorXd::Zero(arm_dof);

	MatrixXd mass_from_robot = MatrixXd::Identity(arm_dof, arm_dof);
    VectorXd coriolis_from_robot = VectorXd::Zero(arm_dof);
	if(!flag_simulation) {
			redis_client.addEigenToReadCallback(0, FRANKA_MASSMATRIX_KEY, mass_from_robot);
			redis_client.addEigenToReadCallback(0, FRANKA_CORIOLIS_KEY, coriolis_from_robot);
	}
	
	if (flag_simulation) {
		arm_robot->updateModel();
		// arm_robot->coriolisForce(arm_coriolis);
	} else {
		arm_robot->updateKinematics();
		arm_robot->_M = mass_from_robot;
		arm_robot->updateInverseInertia();
		arm_coriolis = coriolis_from_robot;
	}

    Vector3d vr_wrist_position = Vector3d::Zero();
	Vector3d vr_wrist_position_center = Vector3d::Zero();
	Matrix3d vr_wrist_orientation = Matrix3d::Identity(3, 3);
	std::string vr_right_trigger, vr_right_grip;

	Vector3d robot_position = Vector3d::Zero();
	Vector3d robot_position_last = Vector3d::Zero();
	Vector3d robot_position_center = Vector3d::Zero();
	Matrix3d robot_orientation = Matrix3d::Identity(3, 3);
	Matrix3d robot_orientation_home = Matrix3d::Identity(3, 3);
    arm_robot->position(robot_position, arm_link_name);
	robot_position_last = robot_position;
	robot_position_center = robot_position;
	arm_robot->rotation(robot_orientation, arm_link_name);
	robot_orientation_home = robot_orientation;

    // Initialize the allegro robot model
	const string fingertip_link_names[] = {"link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"};
    const Vector3d fingertip_pos_in_link = Vector3d(0.0,0.0,0.035);

    // Read wrist position / orientation
    Affine3d T_world_hand = Affine3d::Identity();
    T_world_hand.translation() = robot_position;
    T_world_hand.linear() = robot_orientation;

	auto hand_robot = new Sai2Model::Sai2Model(hand_robot_file, false, T_world_hand);
	int hand_dof = hand_robot->dof();

	if (flag_simulation) {
		hand_robot->_q = q.tail(hand_dof);
		hand_robot->_dq = dq.tail(hand_dof);
	} else {
		hand_robot->_q = redis_client.getEigenMatrixJSON(ALLEGRO_JOINT_ANGLES_KEY);
		hand_robot->_dq = redis_client.getEigenMatrixJSON(ALLEGRO_JOINT_VELOCITIES_KEY);
	}
	VectorXd last_hand_q = hand_robot->_q;
	hand_robot->updateModel();

	// prepare task controller for franka robot with null space joint control
	Eigen::VectorXd arm_posori_torques;
	Eigen::VectorXd arm_joint_torques;

	VectorXd arm_command_torques = VectorXd::Zero(arm_dof);
	Eigen::MatrixXd N_prec = Eigen::MatrixXd::Identity(arm_dof, arm_dof);

	Sai2Primitives::PosOriTask *arm_posori_task = new Sai2Primitives::PosOriTask(arm_robot, arm_link_name, arm_tcp_pos_in_link);
	arm_posori_task->updateTaskModel(N_prec);

	Sai2Primitives::JointTask *arm_joint_task = new Sai2Primitives::JointTask(arm_robot);
	arm_joint_task->updateTaskModel(arm_posori_task->_N);

	arm_joint_task->_desired_position = arm_robot->_q;
	arm_joint_task->_desired_velocity.setZero(arm_robot->_dof);

    // prepare joint hold task for franka robot
    Sai2Primitives::JointTask *arm_hold_task = new Sai2Primitives::JointTask(arm_robot);
	arm_hold_task->updateTaskModel(N_prec);

	arm_hold_task->_desired_position = arm_robot->_q;
	arm_hold_task->_desired_velocity.setZero(arm_robot->_dof);

	// prepare controller for allegro hand
	VectorXd hand_command_torques = VectorXd::Zero(hand_dof);

	MatrixXd hand_N_prec = MatrixXd::Identity(hand_dof, hand_dof);
    MatrixXd hand_N_task_transpose = MatrixXd::Identity(4, 4);
	MatrixXd finger_task_Jacobian = MatrixXd::Zero(3, hand_dof);
    VectorXd hand_q_mid = VectorXd::Zero(hand_dof);
    hand_q_mid << 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5;

	VectorXd finger_current_positions = VectorXd::Zero(4 * 3);
	Vector3d finger_current_position = Vector3d::Zero();

	hand_robot->position(finger_current_position, fingertip_link_names[0], fingertip_pos_in_link);
	finger_current_positions.segment(0, 3) << finger_current_position;
	hand_robot->position(finger_current_position, fingertip_link_names[1], fingertip_pos_in_link);
	finger_current_positions.segment(3, 3) << finger_current_position;
	hand_robot->position(finger_current_position, fingertip_link_names[2], fingertip_pos_in_link);
	finger_current_positions.segment(6, 3) << finger_current_position;
	hand_robot->position(finger_current_position, fingertip_link_names[3], fingertip_pos_in_link);
	// finger_current_position << 0.08, 0.05, -0.02;
	finger_current_positions.segment(9, 3) << finger_current_position;
	
	redis_client.setEigenMatrixJSON(FINGERTIP_POSITION_KEY, finger_current_positions);

	VectorXd finger_target_positions = VectorXd::Zero(4 * 3);

	finger_target_positions = redis_client.getEigenMatrixJSON(FINGERTIP_POSITION_KEY);

	VectorXd one_finger_computed_torques =  VectorXd::Zero(hand_dof); 
	VectorXd all_finger_pos_task_torques = VectorXd::Zero(hand_dof); 

	// create a timer
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(1000); 
	double start_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;

	redis_client.set(CONTROLLER_RUNNING, "1");
	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		double time = timer.elapsedTime() - start_time;

		// read VR keys
		// read wrist tracking params from vr controller
		vr_wrist_position = redis_client.getEigenMatrixJSON(VR_RIGHT_CONTROLLER_POSITION_KEY);
		// vr_wrist_orientation = redis_client.getEigenMatrixJSON(VR_RIGHT_CONTROLLER_ROTATION_KEY);
		vr_right_grip = redis_client.get(VR_RIGHT_CONTROLLER_GRIP_KEY);  // hold to enable haptic control 
		vr_right_trigger = redis_client.get(VR_RIGHT_CONTROLLER_TRIGGER_KEY);  // hold to enable orientation control

		// read control mode from redis
		// std::string control_mode_str = redis_client.get(CONTROL_MODE_KEY);
		if (vr_right_trigger == "0") {
			control_mode = ControlMode::ARM; 
		} else if (vr_right_trigger == "1") {
			control_mode = ControlMode::HAND;
		} else {
			std::cout << " Unknown control mode: " << vr_right_trigger << std::endl;
		}

		// read robot state from redis
		if (flag_simulation){
			q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
			dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);
		}

		if (flag_simulation) {
			arm_robot->_q = q.head(arm_dof);
			arm_robot->_dq = dq.head(arm_dof);
		} else {
			arm_robot->_q = redis_client.getEigenMatrixJSON(FRANKA_JOINT_ANGLES_KEY);
			arm_robot->_dq = redis_client.getEigenMatrixJSON(FRANKA_JOINT_VELOCITIES_KEY);
		}
	
		if(!flag_simulation) {
			redis_client.addEigenToReadCallback(0, FRANKA_MASSMATRIX_KEY, mass_from_robot);
			redis_client.addEigenToReadCallback(0, FRANKA_CORIOLIS_KEY, coriolis_from_robot);
		}

		// Update the state of franka arm
		arm_robot->_q = q.head(arm_dof);
		arm_robot->_dq = dq.head(arm_dof);
		if (flag_simulation) {
			arm_robot->updateModel();
			// arm_robot->coriolisForce(arm_coriolis);
		} else {
			arm_robot->updateKinematics();
			arm_robot->_M = mass_from_robot;
			arm_robot->updateInverseInertia();
			arm_coriolis = coriolis_from_robot;
		}

        // Update the wrist position and orientation
        arm_robot->position(robot_position, arm_link_name);
        arm_robot->rotation(robot_orientation, arm_link_name);

		// Update the state of allegro hand
		if (flag_simulation) {
			hand_robot->_q = q.tail(hand_dof);
			hand_robot->_dq = dq.tail(hand_dof);
		} else {
			hand_robot->_q = redis_client.getEigenMatrixJSON(ALLEGRO_JOINT_ANGLES_KEY);
			hand_robot->_dq = redis_client.getEigenMatrixJSON(ALLEGRO_JOINT_VELOCITIES_KEY);
		}
		hand_robot->updateModel();

		if (control_mode == ControlMode::ARM) {
			// control the franka arm using pos / ori task and allegro had using fixed joint mode 
			arm_command_torques.setZero();
			hand_command_torques.setZero();

			// Update the franka arm controllers
			arm_posori_task->updateTaskModel(N_prec);
			arm_joint_task->updateTaskModel(arm_posori_task->_N);

			// Compute torques for franka robot
			if (vr_right_grip == "1") {
				arm_posori_task->_desired_position = robot_position_center +  R_vr_to_user * (vr_wrist_position - vr_wrist_position_center);
				robot_position_last = robot_position;
			} else {
				arm_posori_task->_desired_position = robot_position_last;
				robot_position_center = robot_position; 
				vr_wrist_position_center = vr_wrist_position;
			}

			arm_posori_task->_desired_velocity =  Eigen::Vector3d::Zero();
			arm_posori_task->_desired_orientation = robot_orientation_home;
			arm_posori_task->_desired_angular_velocity = Eigen::Vector3d::Zero();

			arm_posori_task->computeTorques(arm_posori_torques);
			arm_joint_task->computeTorques(arm_joint_torques);
			arm_command_torques = arm_posori_torques + arm_joint_torques;
			last_arm_q = arm_robot->_q; 

			// Compute finger torques for joint hold task
			hand_command_torques = hand_robot->_M * (-HAND_JOINT_KP * (hand_robot->_q  - last_hand_q) -HAND_JOINT_KV * hand_robot->_dq); 

            // send franka control torques to redis
            redis_client.setEigenMatrixJSON(ARM_TORQUES_COMMANDED_SIM_KEY, arm_command_torques);

            // send allegro hand torques for maintaining position
            redis_client.setEigenMatrixJSON(HAND_TORQUES_COMMANDED_SIM_KEY, hand_command_torques);
		} else if (control_mode == ControlMode::HAND) {
            finger_target_positions = redis_client.getEigenMatrixJSON(FINGERTIP_POSITION_KEY);

			// Switch to joint control of the franka arm and 
			arm_command_torques.setZero();
			hand_command_torques.setZero();

			// compute arm torques for joint hold task
			arm_hold_task->_desired_position = last_arm_q;
			arm_hold_task->_desired_velocity.setZero(arm_robot->_dof);
			arm_hold_task->updateTaskModel(N_prec);
			arm_hold_task->computeTorques(arm_joint_torques);
			arm_command_torques = arm_joint_torques;

			// compute finger torques for position task
			for (int i = 0; i < 4; i++) {
				Vector3d desired_position = Vector3d::Zero();
				Vector3d current_position = Vector3d::Zero();
				Vector3d current_velocity = Vector3d::Zero();
				Vector3d desired_velocity = Vector3d::Zero();
				Vector3d finger_task_force = Vector3d::Zero();
				VectorXd finger_torques =  VectorXd::Zero(4);
				VectorXd posture_torques = VectorXd::Zero(4);
				MatrixXd J_finger = MatrixXd::Zero(3, 4);
				MatrixXd J_bar_finger = MatrixXd::Zero(3, 4);

				desired_position << finger_target_positions.segment(i*3, 3);
				hand_robot->position(current_position, fingertip_link_names[i], fingertip_pos_in_link);
				hand_robot->linearVelocity(current_velocity, fingertip_link_names[i], fingertip_pos_in_link);
				hand_robot->Jv(finger_task_Jacobian, fingertip_link_names[i], fingertip_pos_in_link);
				J_finger = finger_task_Jacobian.block<3, 4>(0, i*4);

				desired_velocity = - HAND_TASK_POSITION_GAIN / HAND_TASK_VELOCITY_GAIN * (current_position - desired_position); 
				if(desired_velocity.norm() > HAND_V_MAX){
					desired_velocity *= HAND_V_MAX / desired_velocity.norm();
				}					
				finger_task_force = - HAND_TASK_VELOCITY_GAIN * (current_velocity - desired_velocity);
				finger_torques = J_finger.transpose() * finger_task_force; 

				J_bar_finger = J_finger.transpose() * (J_finger * J_finger.transpose()).inverse(); 
				hand_N_task_transpose = MatrixXd::Identity(4, 4) - (J_finger.transpose() * J_bar_finger.transpose());
				posture_torques = hand_N_task_transpose * (-HAND_POSTURE_POSITION_GAIN * (hand_robot->_q.segment(i*4, 4) - hand_q_mid.segment(i*4, 4)) - HAND_POSTURE_VELOCITY_GAIN * hand_robot->_dq.segment(i*4, 4));
				
				hand_command_torques.segment(i*4, 4) = (finger_torques + posture_torques); // each pos task generates torques for all joints, with only the relevant finger joints being nonzero					
			}
			last_hand_q = hand_robot->_q;
		} else {
			arm_command_torques.setZero();
			hand_command_torques.setZero();
		}

		// publish wrist position and orientation for allegro controller
		redis_client.setEigenMatrixJSON(ALLEGRO_PALM_ORIENTATION, robot_orientation);

		if (flag_simulation) {
			// send franka control torques to redis
			redis_client.setEigenMatrixJSON(ARM_TORQUES_COMMANDED_SIM_KEY, arm_command_torques);
			// send allegro control torques to redis
			redis_client.setEigenMatrixJSON(HAND_TORQUES_COMMANDED_SIM_KEY, hand_command_torques);
		} else {
			// send franka control torques to redis
			redis_client.setEigenMatrixJSON(FRANKA_COMMAND_TORQUES_KEY, arm_command_torques);
			// send allegro control torques to redis
			redis_client.setEigenMatrixJSON(ALLEGRO_COMMAND_TORQUES_KEY, hand_command_torques);
		}
		
        controller_counter++;
	}

	arm_command_torques.setZero();
	redis_client.setEigenMatrixJSON(ARM_TORQUES_COMMANDED_SIM_KEY, arm_command_torques);
	redis_client.setEigenMatrixJSON(HAND_TORQUES_COMMANDED_SIM_KEY, hand_command_torques);
	redis_client.set(CONTROLLER_RUNNING, "0");

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
    std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";


	return 0;
}
