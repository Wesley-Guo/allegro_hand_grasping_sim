// This example tests the haptic device driver and the open-loop bilateral teleoperation controller.

#include "Sai2Model.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "tasks/JointTask.h"
#include "tasks/PosOriTask.h"
// #include "filters/ButterworthFilter.h"
#include "../src/Logger.h"
// #include "perception/ForceSpaceParticleFilter.h"

#include <iostream>
#include <string>
#include <random>
#include <queue>

#define INIT            0
#define CONTROL         1

#include <signal.h>
bool runloop = false;
void sighandler(int){runloop = false;}

using namespace std;
using namespace Eigen;

const string robot_file = "./resources/panda_arm.urdf";
const string hand_file = "./resources/allegro_hand.urdf";

// redis keys:
// robot local control loop
string JOINT_ANGLES_KEY = "sai2::AllegroGraspSim::00::simviz::sensors::q";
string JOINT_VELOCITIES_KEY = "sai2::AllegroGraspSim::00::simviz::sensors::dq";
string ROBOT_COMMAND_TORQUES_KEY = "sai2::AllegroGraspSim::00::simviz::actuators::tau_cmd";
// string ROBOT_SENSED_FORCE_KEY = "sai2::AllegroGraspSim::00::simviz::sensors::sensed_force";

string HAND_POSITION_COMMANDED = "allegroHand::controller::joint_positions_commanded";
string HAND_STATES_COMMANDED = "allegroHand::controller::joint_states_commanded";

RedisClient redis_client;


const double control_loop_freq = 1000.0;

// set control link and point for posori task
const string robot_ee_link_name = "end_effector";
const Vector3d robot_ee_pos_in_link = Vector3d(0.0,0.0,0.035);

// set sensor frame transform in end-effector frame
// Affine3d sensor_transform_in_link = Affine3d::Identity();
// const Vector3d sensor_pos_in_link = Eigen::Vector3d(0.0,0.0,0.034);

// const bool flag_simulation = false;
const bool flag_simulation = true;

int main() {

	// if(!flag_simulation) {
	// 	ROBOT_COMMAND_TORQUES_KEY = "sai2::FrankaPanda::actuators::fgc";
	// 	JOINT_ANGLES_KEY  = "sai2::FrankaPanda::sensors::q";
	// 	JOINT_VELOCITIES_KEY = "sai2::FrankaPanda::sensors::dq";
	// 	MASSMATRIX_KEY = "sai2::FrankaPanda::sensors::model::massmatrix";
	// 	CORIOLIS_KEY = "sai2::FrankaPanda::sensors::model::coriolis";
	// 	ROBOT_SENSED_FORCE_KEY = "sai2::ATIGamma_Sensor::force_torque";
	// }

	// start redis client local
	redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load panda robot model
	Affine3d T_world_robot = Affine3d::Identity();
	T_world_robot.translation() = Vector3d(0, 0, 0);
	auto robot = new Sai2Model::Sai2Model(robot_file, false, T_world_robot);

	robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	robot->updateModel();

	int robot_dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(robot_dof);
	VectorXd robot_coriolis = VectorXd::Zero(robot_dof);
	int state = INIT;
	MatrixXd N_prec = MatrixXd::Identity(robot_dof,robot_dof);

	// load allegro hand model
	auto hand = new Sai2Model::Sai2Model(hand_file, false); //Transform between hand and world will be updated later
	hand->updateModel();

	// hand state initialization
	redis_client.set(HAND_STATES_COMMANDED, "DEFAULT");

	// joint task
	auto joint_task = new Sai2Primitives::JointTask(robot);
	VectorXd joint_task_torques = VectorXd::Zero(robot_dof);
	joint_task->_use_interpolation_flag = true;
	joint_task->_use_velocity_saturation_flag = false;

	joint_task->_kp = 200.0;
	joint_task->_kv = 25.0;
	joint_task->_ki = 50.0;

	Eigen::VectorXd g(robot_dof); //joint space gravity vector


    // VectorXd q_init(dof);
    // q_init << 0, -30, 0, -130, 0, 100, 0;
    // q_init *= M_PI/180.0;
    joint_task->_desired_position = robot->_q; // use current robot config as init config

	// posori task
	// const string link_name = "end_effector";
	// const Vector3d pos_in_link = Vector3d(0, 0, 0.12);
	auto posori_task = new Sai2Primitives::PosOriTask(robot, robot_ee_link_name, robot_ee_pos_in_link);
	Vector3d x_init = posori_task->_current_position;
	Matrix3d R_init = posori_task->_current_orientation;
	// compute expected default rotation to send to haptic
    // robot->_q = q_init;
    // robot->updateModel();
	Matrix3d R_default = Matrix3d::Identity();
	Vector3d pos_default = Vector3d::Zero();
	robot->rotation(R_default, robot_ee_link_name);
	robot->position(pos_default, robot_ee_link_name, robot_ee_pos_in_link);

	// cout << "R default:\n" << R_default << endl; 

	VectorXd posori_task_torques = VectorXd::Zero(robot_dof);
	posori_task->_use_interpolation_flag = true;

	posori_task->_otg->setMaxLinearVelocity(0.30);
	posori_task->_otg->setMaxLinearAcceleration(1.0);
	posori_task->_otg->setMaxLinearJerk(5.0);

	posori_task->_otg->setMaxAngularVelocity(M_PI/1.5);
	posori_task->_otg->setMaxAngularAcceleration(3*M_PI);
	posori_task->_otg->setMaxAngularJerk(15*M_PI);

	posori_task->_kp_pos = 100.0;
	posori_task->_kv_pos = 17.0;

	posori_task->_kp_ori = 200.0;
	posori_task->_kv_ori = 23.0;

	// if(!flag_simulation) {
	// 	force_bias << 1.50246,   -8.19902,  -0.695169,  -0.987652,   0.290632, -0.0453239;
	// 	tool_mass = 0.33;
	// 	tool_com = Vector3d(-0.00492734, -0.00295005,   0.0859595);
	// }

	VectorXd q_des_hand = VectorXd::Zero(16);
	string activeStateString;

	// setup redis keys to be updated with the callback
	redis_client.createReadCallback(0);
	redis_client.createWriteCallback(0);

	// Objects to read from redis
    redis_client.addEigenToReadCallback(0, JOINT_ANGLES_KEY, robot->_q);
    redis_client.addEigenToReadCallback(0, JOINT_VELOCITIES_KEY, robot->_dq);
	
	redis_client.addStringToReadCallback(0, HAND_STATES_COMMANDED, activeStateString);

	// Objects to write to redis
	redis_client.addEigenToWriteCallback(0, ROBOT_COMMAND_TORQUES_KEY, command_torques);
	redis_client.addEigenToWriteCallback(0, HAND_POSITION_COMMANDED, q_des_hand);

	// setup data logging
	string folder = "../../00-ball_reorient/data_logging/data/";
	string filename = "data";
    auto logger = new Logging::Logger(100000, folder + filename);
	
	Vector3d log_robot_ee_position = x_init;
	Vector3d log_robot_ee_velocity = Vector3d::Zero();
	VectorXd log_joint_angles = robot->_q;
	VectorXd log_joint_velocities = robot->_dq;
	VectorXd log_joint_command_torques = command_torques;
    Vector3d log_desired_force = Vector3d::Zero();

	logger->addVectorToLog(&log_robot_ee_position, "robot_ee_position");
	logger->addVectorToLog(&log_robot_ee_velocity, "robot_ee_velocity");
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
	double prev_time = 0;
	// double dt = 0;
	bool fTimerDidSleep = true;
	double start_time = timer.elapsedTime(); //secs

	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		current_time = timer.elapsedTime() - start_time;

		// read haptic state and robot state
		redis_client.executeReadCallback(0);

		if(flag_simulation) {
			robot->updateModel();
			robot->coriolisForce(robot_coriolis);
		}
		else {
			robot->updateKinematics();
			// robot->_M = mass_from_robot;
			robot->updateInverseInertia();
			// coriolis = coriolis_from_robot;
		}

		N_prec.setIdentity(robot_dof,robot_dof);

		posori_task->updateTaskModel(N_prec);
		N_prec = posori_task->_N;

		joint_task->updateTaskModel(N_prec);

		if(state == INIT) {
			robot->gravityVector(g);
			joint_task->updateTaskModel(MatrixXd::Identity(robot_dof,robot_dof));

			joint_task->computeTorques(joint_task_torques);
			command_torques = joint_task_torques + robot_coriolis;
			// command_torques = joint_task_torques + coriolis + g;

			if((joint_task->_desired_position - joint_task->_current_position).norm() < 0.2) {
				// Reinitialize controllers
				posori_task->reInitializeTask();
				joint_task->reInitializeTask();

				joint_task->_kp = 50.0;
				joint_task->_kv = 13.0;
				joint_task->_ki = 0.0;

				state = CONTROL;
			}
		}

		else if(state == CONTROL) {

			try	{
				posori_task->computeTorques(posori_task_torques);
			}
			catch(exception e) {
				cout << "control cycle: " << controller_counter << endl;
				cout << "error in the torque computation of posori_task:" << endl;
				cerr << e.what() << endl;
				cout << "setting torques to zero for this control cycle" << endl;
				cout << endl;
				// posori_task_torques.setZero(); // set task torques to zero, TODO: test this
			}
			joint_task->computeTorques(joint_task_torques);

			command_torques = posori_task_torques + joint_task_torques + robot_coriolis;
		}

		// if (activeStateString == "PREGRASP")
		// 	q_des_hand << 0.11, 0.34, 1.3, 1, 0.05, 0.43, 1.1, 1.3, -0.051, 0.46, 0.98, 1.2, 1.4, 0.075, 0.04, 1.6;
		// if (activeStateString == "TOUCH")
		// 	q_des_hand << 0.12, 0.53, 1.4, 0.88, 0.15, 0.67, 1.1, 1.1, -0.047, 0.46, 0.98, 1.2, 1.4, 0.076, 0.24, 1.3;
		// if (activeStateString == "GRASP")
		// 	q_des_hand << 0.062, 0.52, 1.4, 0.88, 0.21, 0.66, 1.1, 1.1, -0.052, 0.46, 0.98, 1.2,  1.4, 0.1, 0.45, 1.2;

		if (activeStateString == "PREGRASP")
			q_des_hand << 0.11, 0.34, 1.3, 1, 0.05, 0.43, 1.1, 1.3, -0.051, 0.46, 0.98, 1.2, 0, 1.2, 0.8, 1.6;
		else if (activeStateString == "TOUCH")
			q_des_hand << 0.12, 0.53, 1.4, 0.88, 0.15, 0.67, 1.1, 1.1, -0.047, 0.46, 0.98, 1.2, 0, 1.2, 0.8, 1.3;
		else if (activeStateString == "GRASP")
			q_des_hand << 0.062, 0.52, 1.4, 0.88, 0.21, 0.66, 1.1, 1.1, -0.052, 0.46, 0.98, 1.2,  0, 1.3, 0.8, 1.2;
		else
			q_des_hand << 0.11, 0.34, 1.3, 1, 0.05, 0.43, 1.1, 1.3, -0.051, 0.46, 0.98, 1.2, 0, 1.2, 0.8, 1.6;

		// write control torques
		redis_client.executeWriteCallback(0);

		// update logger values
		Vector3d ee_pos = Vector3d::Zero();
		Vector3d ee_vel = Vector3d::Zero();
		robot->position(ee_pos, robot_ee_link_name, robot_ee_pos_in_link);
		robot->linearVelocity(ee_vel, robot_ee_link_name, robot_ee_pos_in_link);
		
		log_robot_ee_position = ee_pos;
		log_robot_ee_velocity = ee_vel;
		log_joint_angles = robot->_q;
		log_joint_velocities = robot->_dq;
		log_joint_command_torques = command_torques;
        log_desired_force = posori_task->_desired_force;
	
		controller_counter++;
	}

	// stop logger
	logger->stop();

	//// Send zero force/torque to robot ////
	command_torques.setZero();
	redis_client.setEigenMatrixJSON(ROBOT_COMMAND_TORQUES_KEY, command_torques);


	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
	std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}
