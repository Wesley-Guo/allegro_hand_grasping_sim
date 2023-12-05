// This example tests the haptic device driver and the open-loop bilateral teleoperation controller.

#include "Sai2Model.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "tasks/JointTask.h"
#include "tasks/PositionTask.h"
#include "tasks/PosOriTask.h"
#include "Sai2Primitives.h"
// #include "filters/ButterworthFilter.h"
#include "../src/Logger.h"
// #include "perception/ForceSpaceParticleFilter.h"

#include <iostream>
#include <string>
#include <random>
#include <queue>
#include <boost/algorithm/clamp.hpp>

#define INIT            0
#define CONTROL         1

#include <signal.h>
bool runloop = false;
void sighandler(int){runloop = false;}

using namespace std;
using namespace Eigen;

const string robot_file = "./resources/allegro_hand.urdf";

// redis keys:
// robot local control loop
string JOINT_ANGLES_SIM_KEY = "sai2::panda_robot_with_allegro::sensors::q";
string JOINT_VELOCITIES_SIM_KEY = "sai2::panda_robot_with_allegro::sensors::dq";
string JOINT_ANGLES_DESIRED_SIM_KEY = "sai2::panda_robot_with_allegro::control::q_des";
string ROBOT_COMMAND_TORQUES_SIM_KEY = "sai2::allegro::actuators::tau";

const std::string PALM_POSITION_KEY = "sai2::panda_robot_with_allegro::sensors::palm_position";
const std::string PALM_ORINETATION_KEY = "sai2::panda_robot_with_allegro::sensors::palm_orientation";

string CONTROL_MODE = "allegroHand::controller::control_mode";
string JOINT_ANGLES_KEY = "allegroHand::sensors::joint_positions";
string JOINT_VELOCITIES_KEY = "allegroHand::sensors::joint_velocities";
string JOINT_ANGLES_DESIRED_KEY = "allegroHand::controller::joint_positions_commanded";
string ROBOT_COMMAND_TORQUES_KEY = "allegroHand::controller::joint_torques_commanded";
string FINGERTIP_POSITION_KEY = "allegroHand::controller::finger_positions_commanded";

RedisClient redis_client;


const double control_loop_freq = 1000.0;

// set control link and point for posori tasks
const string fingertip_link_names[] = {"link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"};
const Vector3d fingertip_pos_in_link = Vector3d(0.0,0.0,0.035);

// const bool flag_simulation = false;
const bool flag_simulation = true;

int main() {
	// start redis client local
	redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load allegro hand model
	Affine3d T_world_robot = Affine3d::Identity();
	T_world_robot.translation() = redis_client.getEigenMatrixJSON(PALM_POSITION_KEY);
	T_world_robot.linear() = redis_client.getEigenMatrixJSON(PALM_ORINETATION_KEY);

	auto robot = new Sai2Model::Sai2Model(robot_file, false, T_world_robot);
	int robot_dof = robot->dof();
	VectorXd q = VectorXd::Zero(7 + robot_dof);
	VectorXd dq = VectorXd::Zero(7 + robot_dof);
	VectorXd tau = VectorXd::Zero(7 + robot_dof);
	VectorXd gravity = VectorXd::Zero(robot_dof);

	Vector3d palm_position = Vector3d::Zero();
	Matrix3d R_palm = Matrix3d::Identity(3, 3);

	if (flag_simulation){
		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
		robot->_q = q.tail(robot_dof);
		dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);
		robot->_dq = dq.tail(robot_dof);
	} else {
		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
		robot->_q = q.tail(robot_dof);
		dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
		robot->_dq = dq.tail(robot_dof);
	}
	robot->updateModel();

	VectorXd command_torques = VectorXd::Zero(robot_dof);
	VectorXd robot_coriolis = VectorXd::Zero(robot_dof);
	MatrixXd N_prec = MatrixXd::Identity(robot_dof, robot_dof);
    MatrixXd N_task_transpose = MatrixXd::Identity(4, 4);
	MatrixXd finger_task_Jacobian = MatrixXd::Zero(3, robot_dof);
    VectorXd q_mid = VectorXd::Zero(robot_dof);
    q_mid << 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5;

	// controller_state
	int state = INIT;

	// joint task for initialization
	auto joint_task = new Sai2Primitives::JointTask(robot);
	joint_task->_use_interpolation_flag = false;
	joint_task->_use_velocity_saturation_flag = true;

	VectorXd joint_task_torques = VectorXd::Zero(robot_dof);
	joint_task->_kp = 150.0;
	joint_task->_kv = 25.0;
	joint_task->_ki = 0.0;


	double V_MAX, TASK_POSITION_GAIN, TASK_VELOCITY_GAIN, POSTURE_POSITION_GAIN, POSTURE_VELOCITY_GAIN;
	if (flag_simulation){
		V_MAX = 1.0;

		TASK_POSITION_GAIN = 125.0;
		TASK_VELOCITY_GAIN = 2.25;

		POSTURE_POSITION_GAIN = 0.0025;
		POSTURE_VELOCITY_GAIN = 0.0015;
	} else {
		V_MAX = 1.5;

		TASK_POSITION_GAIN = 175.0;
		TASK_VELOCITY_GAIN = 2.0;

		POSTURE_POSITION_GAIN = 0.05;
	    POSTURE_VELOCITY_GAIN = 0.025;

	}

	Eigen::VectorXd g(robot_dof); //joint space gravity vector
    joint_task->_desired_position = robot->_q; // use current robot config as init config
	if (flag_simulation) {
		redis_client.setEigenMatrixJSON(JOINT_ANGLES_DESIRED_SIM_KEY, robot->_q);
	} else {
		redis_client.setEigenMatrixJSON(JOINT_ANGLES_DESIRED_KEY, robot->_q);
	}
	
	VectorXd finger_current_positions = VectorXd::Zero(4 * 3);
	Vector3d finger_current_position = Vector3d::Zero();

	robot->position(finger_current_position, fingertip_link_names[0], fingertip_pos_in_link);
	finger_current_positions.segment(0, 3) << finger_current_position;
	robot->position(finger_current_position, fingertip_link_names[1], fingertip_pos_in_link);
	finger_current_positions.segment(3, 3) << finger_current_position;
	robot->position(finger_current_position, fingertip_link_names[2], fingertip_pos_in_link);
	finger_current_positions.segment(6, 3) << finger_current_position;
	robot->position(finger_current_position, fingertip_link_names[3], fingertip_pos_in_link);
	finger_current_position << 0.08, 0.05, -0.02;
	finger_current_positions.segment(9, 3) << finger_current_position;
	
	redis_client.setEigenMatrixJSON(FINGERTIP_POSITION_KEY, finger_current_positions);

	VectorXd finger_target_positions = VectorXd::Zero(4 * 3);

	finger_target_positions = redis_client.getEigenMatrixJSON(FINGERTIP_POSITION_KEY);

	VectorXd one_finger_computed_torques =  VectorXd::Zero(robot_dof); 
	VectorXd all_pos_task_torques = VectorXd::Zero(robot_dof); 

	redis_client.createReadCallback(0);
	
	if (flag_simulation){
		// Objects to read from redis
		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
		robot->_q = q.tail(robot_dof);
		dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);
		robot->_dq = dq.tail(robot_dof);

		redis_client.addEigenToReadCallback(0, PALM_POSITION_KEY, palm_position);
		redis_client.addEigenToReadCallback(0, PALM_ORINETATION_KEY, R_palm);
	} else {
		// Objects to read from redis
		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
		robot->_q = q.tail(robot_dof);
		dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
		robot->_dq = dq.tail(robot_dof);

		redis_client.addEigenToReadCallback(0, PALM_POSITION_KEY, palm_position);
		redis_client.addEigenToReadCallback(0, PALM_ORINETATION_KEY, R_palm);
	}

	// setup data logging
	string folder = "";
	string filename = "data";
    auto logger = new Logging::Logger(10000, folder + filename);
	
    Vector3d finger_tip_pos  = Vector3d::Zero();
    Vector3d finger_tip_vel  = Vector3d::Zero();
    robot->position(finger_tip_pos, fingertip_link_names[0], fingertip_pos_in_link);
    robot->linearVelocity(finger_tip_vel, fingertip_link_names[0], fingertip_pos_in_link);

	Vector3d log_finger_tip_position = finger_tip_pos;
	Vector3d log_finger_tip_velocity = Vector3d::Zero();
	VectorXd log_joint_angles = robot->_q;
	VectorXd log_joint_velocities = robot->_dq;
	VectorXd log_joint_command_torques = command_torques;
    Vector3d log_desired_force = Vector3d::Zero();

	logger->addVectorToLog(&log_finger_tip_position, "log_finger_tip_position");
	logger->addVectorToLog(&log_finger_tip_velocity, "log_finger_tip_velocity");
	logger->addVectorToLog(&log_joint_angles, "joint_angles");
	logger->addVectorToLog(&log_joint_velocities, "joint_velocities");
	logger->addVectorToLog(&log_joint_command_torques, "joint_command_torques");
    logger->addVectorToLog(&log_desired_force, "desired_force");

	logger->start();

	// create a timer
	runloop = true;
	unsigned long long controller_counter = 0;
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(control_loop_freq); //Compiler en mode release
	double current_time = 0;
	// double dt = 0;
	bool fTimerDidSleep = true;
	double start_time = timer.elapsedTime(); //secs

	int num_iters = 0;

	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		current_time = timer.elapsedTime() - start_time;

		finger_target_positions = redis_client.getEigenMatrixJSON(FINGERTIP_POSITION_KEY);

		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
		robot->_q = q.tail(robot_dof);
		dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);
		robot->_dq = dq.tail(robot_dof);
		
		robot->updateModel();

		// Set Task Hirearchy
		N_prec.setIdentity(robot_dof,robot_dof);
		num_iters += 1;

		if(state == INIT) {
			joint_task->_desired_position = q_mid;
            joint_task->updateTaskModel(N_prec);
			joint_task->computeTorques(joint_task_torques);
			command_torques = joint_task_torques;

			std::cout << " joint mode " << std::endl;

			if (num_iters > 500){
				state = CONTROL;
			}
			
		}

		if(state == CONTROL) {
			all_pos_task_torques = VectorXd::Zero(robot_dof);

			try	{
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
					robot->position(current_position, fingertip_link_names[i], fingertip_pos_in_link);
    				robot->linearVelocity(current_velocity, fingertip_link_names[i], fingertip_pos_in_link);
					robot->Jv(finger_task_Jacobian, fingertip_link_names[i], fingertip_pos_in_link);
					J_finger = finger_task_Jacobian.block<3, 4>(0, i*4);

					std::cout << " J_finger " << J_finger  << std::endl;

					desired_velocity = - TASK_POSITION_GAIN / TASK_VELOCITY_GAIN * (current_position - desired_position); 
					if(desired_velocity.norm() > V_MAX){
						desired_velocity *= V_MAX / desired_velocity.norm();
					}					
					finger_task_force = - TASK_VELOCITY_GAIN * (current_velocity - desired_velocity);
					finger_torques = J_finger.transpose() * finger_task_force; 

					J_bar_finger = J_finger.transpose() * (J_finger * J_finger.transpose()).inverse(); 
					std::cout << " J_bar_finger " << J_bar_finger  << std::endl;
					N_task_transpose = MatrixXd::Identity(4, 4) - (J_finger.transpose() * J_bar_finger.transpose());
					std::cout << " N_task_transpose " << N_task_transpose  << std::endl;
					posture_torques = N_task_transpose * (-POSTURE_POSITION_GAIN * (robot->_q.segment(i*4, 4) - q_mid.segment(i*4, 4)) - POSTURE_VELOCITY_GAIN * robot->_dq.segment(i*4, 4));

					std::cout << " finger: " << i << " finger torques: " << finger_torques << " posture torques: " << posture_torques << std::endl;
					
					all_pos_task_torques.segment(i*4, 4) = (finger_torques + posture_torques); // each pos task generates torques for all joints, with only the relevant finger joints being nonzero					
				}
			}
			catch(exception e) {
				cout << "control cycle: " << controller_counter << endl;
				cout << "error in the torque computation of finger_pos_task:" << endl;
				cerr << e.what() << endl;
				cout << "setting torques to zero for this control cycle" << endl;
				cout << endl;
			}


			command_torques = all_pos_task_torques;	
		}


		std::cout << "  command torques: " << command_torques << std::endl;
		// write control torques
		redis_client.setEigenMatrixJSON(ROBOT_COMMAND_TORQUES_SIM_KEY, command_torques);

		if (num_iters > 20){
			redis_client.setEigenMatrixJSON(ROBOT_COMMAND_TORQUES_SIM_KEY, command_torques.setZero());
			std::exit(0);
		}

		// update logger values
		robot->position(finger_tip_pos, fingertip_link_names[0], fingertip_pos_in_link);
		robot->linearVelocity(finger_tip_vel, fingertip_link_names[0], fingertip_pos_in_link);
		
		log_finger_tip_position = finger_tip_pos;
		log_finger_tip_velocity = finger_tip_vel;
		log_joint_angles = robot->_q;
		log_joint_velocities = robot->_dq;
		log_joint_command_torques = command_torques;
        log_desired_force = Vector3d::Zero();
	
		controller_counter++;
	}

	// stop logger
	logger->stop();

	//// Send zero force/torque to robot ////
	command_torques.setZero();
	// write control torques
	tau = redis_client.getEigenMatrixJSON(ROBOT_COMMAND_TORQUES_SIM_KEY);
	tau.tail(robot_dof) = command_torques;
	redis_client.setEigenMatrixJSON(ROBOT_COMMAND_TORQUES_SIM_KEY, tau);


	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
	std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}
