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
const std::string JOINT_ANGLES_SIM_KEY = "sai2::panda_robot_with_allegro::sensors::q";
const std::string JOINT_VELOCITIES_SIM_KEY = "sai2::panda_robot_with_allegro::sensors::dq";

// allegro 
std::string ALLEGRO_JOINT_ANGLES_KEY = "allegroHand::sensors::joint_positions";
std::string ALLEGRO_JOINT_VELOCITIES_KEY = "allegroHand::sensors::joint_velocities";

std::string FINGERTIP_POSITION_KEY = "allegroHand::controller::finger_positions_commanded";

const std::string WRIST_POSITION_KEY = "mocap::right_hand::position";
const std::string WRIST_ORIENTATION_KEY = "mocap::right_hand::orientation";

// values this key can take are "arm" or "hand"
const string CONTROL_MODE_KEY = "sai2::panda_robot_with_allegro::control_mode";

// - write
const std::string ARM_TORQUES_COMMANDED_SIM_KEY = "sai2::panda_robot::actuators::tau";
const std::string HAND_TORQUES_COMMANDED_SIM_KEY = "sai2::allegro::actuators::tau";

std::string ALLEGRO_TORQUES_COMMANDED_KEY = "allegroHand::controller::joint_torques_commanded";

const std::string CONTROLLER_RUNNING = "sai2::panda_robot::controller_running";

unsigned long long controller_counter = 0;


// gain values for allegro
const double HAND_JOINT_KP = 125.0;
const double HAND_JOINT_KV = 45.0;

const double HAND_V_MAX = 1.0;

const double HAND_TASK_POSITION_GAIN = 125.0;
const double HAND_TASK_VELOCITY_GAIN = 12.25;

const double HAND_POSTURE_POSITION_GAIN = 0.025;
const double HAND_POSTURE_VELOCITY_GAIN = 0.015;

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
	ControlMode control_mode = ControlMode::HAND;

    // Initialize the allegro robot model
	const string fingertip_link_names[] = {"link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"};
    const Vector3d fingertip_pos_in_link = Vector3d(0.0,0.0,0.035);

	auto hand_robot = new Sai2Model::Sai2Model(hand_robot_file, false);
	int hand_dof = hand_robot->dof();

    // Initialize franka robot model
	const string arm_link_name = "link7";
	const Vector3d arm_tcp_pos_in_link = Vector3d(0, 0, 0.1);

	auto arm_robot = new Sai2Model::Sai2Model(arm_robot_file, false);
	const int arm_dof = arm_robot->dof();

	// Read robot states and update the robot model
	VectorXd q = VectorXd::Zero(arm_dof + hand_dof);
	VectorXd dq = VectorXd::Zero(arm_dof + hand_dof);

	q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
	dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);
	arm_robot->_q = q.head(arm_dof);
	arm_robot->_dq = dq.head(arm_dof);

	VectorXd last_arm_q = arm_robot->_q;
	VectorXd initial_arm_q = arm_robot->_q;
	arm_robot->updateModel();

	hand_robot->_q = q.tail(hand_dof);
	hand_robot->_dq = dq.tail(hand_dof);
	VectorXd last_hand_q = hand_robot->_q;
	hand_robot->updateModel();

	// Read wrist position and orientation
	Vector3d x_d = Vector3d::Zero();
	arm_robot->position(x_d, arm_link_name, arm_tcp_pos_in_link);

	Eigen::Matrix3d R_fixed =  Eigen::Matrix3d::Identity();
	R_fixed << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    arm_robot->rotation(R_fixed, arm_link_name);

	// prepare controller for franka robot
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
	finger_current_position << 0.08, 0.05, -0.02;
	finger_current_positions.segment(9, 3) << finger_current_position;
	
	redis_client.setEigenMatrixJSON(FINGERTIP_POSITION_KEY, finger_current_positions);

	VectorXd finger_target_positions = VectorXd::Zero(4 * 3);

	finger_target_positions = redis_client.getEigenMatrixJSON(FINGERTIP_POSITION_KEY);

	VectorXd one_finger_computed_torques =  VectorXd::Zero(hand_dof); 
	VectorXd all_pos_task_torques = VectorXd::Zero(hand_dof); 


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

		// read control mode from redis
		std::string control_mode_str = redis_client.get(CONTROL_MODE_KEY);
		if (control_mode_str == "arm") {
			control_mode = ControlMode::ARM; 
		} else if (control_mode_str == "hand") {
			control_mode = ControlMode::HAND;
		} else {
			std::cout << " Unknown control mode: " << control_mode_str << std::endl;
		}

		// read robot state from redis
		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
		dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);

		// Update the state of franka arm
		arm_robot->_q = q.head(arm_dof);
		arm_robot->_dq = dq.head(arm_dof);
		arm_robot->updateModel();

		// Update the state of allegro hand
		hand_robot->_q = q.tail(hand_dof);
		hand_robot->_dq = dq.tail(hand_dof);
		hand_robot->updateModel();

		if (control_mode == ControlMode::ARM) {
			// control the franka arm using pos / ori task and allegro had using fixed joint mode 
			arm_command_torques.setZero();
			hand_command_torques.setZero();

			// Update the franka arm controllers
			arm_posori_task->updateTaskModel(N_prec);
			arm_joint_task->updateTaskModel(arm_posori_task->_N);

			// Compute torques for franka robot
			arm_posori_task->_desired_orientation = R_fixed;
			arm_posori_task->_desired_angular_velocity = Eigen::Vector3d::Zero();

			arm_posori_task->_desired_velocity =  Eigen::Vector3d::Zero();
			x_d = redis_client.getEigenMatrixJSON(WRIST_POSITION_KEY);
			arm_posori_task->_desired_position = x_d;

			arm_posori_task->computeTorques(arm_posori_torques);
			arm_joint_task->computeTorques(arm_joint_torques);
			arm_command_torques = arm_posori_torques + arm_joint_torques;
			last_arm_q = hand_robot->_q; 

			// Compute finger torques for joint hold task
			hand_command_torques = hand_robot->_M * (-HAND_JOINT_KP * (hand_robot->_q  - last_hand_q) -HAND_JOINT_KV * hand_robot->_dq); 
		} else if (control_mode == ControlMode::HAND) {
			// Switch to joint control of the franka arm and 
			arm_command_torques.setZero();
			hand_command_torques.setZero();

			// compute arm torques for joint hold task
			arm_joint_task->_desired_position = last_arm_q;
			arm_joint_task->_desired_velocity.setZero(arm_robot->_dof);
			arm_joint_task->updateTaskModel(N_prec);
			arm_joint_task->computeTorques(arm_joint_torques);
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
		// send franka control torques to redis
		redis_client.setEigenMatrixJSON(ARM_TORQUES_COMMANDED_SIM_KEY, arm_command_torques);

		// send allegro hand torques for maintaining position
		redis_client.setEigenMatrixJSON(HAND_TORQUES_COMMANDED_SIM_KEY, hand_command_torques);

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
