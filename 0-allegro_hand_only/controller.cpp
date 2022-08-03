// This example tests the haptic device driver and the open-loop bilateral teleoperation controller.

#include "Sai2Model.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "tasks/JointTask.h"
#include "tasks/PositionTask.h"
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

const string robot_file = "./resources/allegro_hand.urdf";

// redis keys:
// robot local control loop
string JOINT_ANGLES_KEY = "sai2::AllegroGraspSim::0::simviz::sensors::q";
string JOINT_VELOCITIES_KEY = "sai2::AllegroGraspSim::0::simviz::sensors::dq";
string JOINT_ANGLES_DESIRED_KEY = "sai2::AllegroGraspSim::0::simviz::control::q_des";
string ROBOT_COMMAND_TORQUES_KEY = "sai2::AllegroGraspSim::0::simviz::actuators::tau_cmd";

RedisClient redis_client;


const double control_loop_freq = 1000.0;

// set control link and point for posori tasks
const string hand_ee_link_names[] = {"link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip"};
const Vector3d hand_ee_pos_in_link = Vector3d(0.0,0.0,0.035);


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

	// load allegro hand model
	Affine3d T_world_robot = Affine3d::Identity();
	T_world_robot.translation() = Vector3d(0.35, 0.0, 0.6);
	auto robot = new Sai2Model::Sai2Model(robot_file, false, T_world_robot);
	// auto robot = new Sai2Model::Sai2Model(robot_file, false);

	robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
	robot->updateModel();

	int robot_dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(robot_dof);
	VectorXd robot_coriolis = VectorXd::Zero(robot_dof);
	MatrixXd N_prec = MatrixXd::Identity(robot_dof,robot_dof);
	MatrixXd finger_task_Jacobian = MatrixXd::Zero(3,robot_dof);
	MatrixXd combined_task_Jacobian = MatrixXd::Zero(int(3*4),robot_dof);
	// cout<<"combined task Jacobian" << endl << combined_task_Jacobian  << endl;


	// controller_state
	int state = INIT;

	// joint task
	auto joint_task = new Sai2Primitives::JointTask(robot);
	joint_task->_use_interpolation_flag = false;
	joint_task->_use_velocity_saturation_flag = true;
	joint_task->setDynamicDecouplingFull();

	VectorXd joint_task_torques = VectorXd::Zero(robot_dof);
	joint_task->_kp = 50.0;
	joint_task->_kv = 13.0;
	joint_task->_ki = 0.0;

	Eigen::VectorXd g(robot_dof); //joint space gravity vector
    joint_task->_desired_position = robot->_q; // use current robot config as init config
	redis_client.setEigenMatrixJSON(JOINT_ANGLES_DESIRED_KEY, robot->_q);

	// define posori tasks for each fingertip
	const string link_name = "end_effector";
	const Vector3d pos_in_link = Vector3d(0, 0, 0.12);
	auto finger_pos_task_0 = new Sai2Primitives::PositionTask(robot, hand_ee_link_names[0], hand_ee_pos_in_link);
	auto finger_pos_task_1 = new Sai2Primitives::PositionTask(robot, hand_ee_link_names[1], hand_ee_pos_in_link);
	auto finger_pos_task_2 = new Sai2Primitives::PositionTask(robot, hand_ee_link_names[2], hand_ee_pos_in_link);
	auto finger_pos_task_3 = new Sai2Primitives::PositionTask(robot, hand_ee_link_names[3], hand_ee_pos_in_link);

	Sai2Primitives::PositionTask* finger_pos_tasks[] = {finger_pos_task_0, finger_pos_task_1, finger_pos_task_2, finger_pos_task_3};

	//For each fingertip posori task, initialize control parameters
	for (int i = 0; i < 4; i++) {
		finger_pos_tasks[i]->_use_interpolation_flag = true;
		finger_pos_tasks[i]->setDynamicDecouplingFull();
		// finger_pos_tasks[i]->_otg->setMaxLinearVelocity(0.30);
		// finger_pos_tasks[i]->_otg->setMaxLinearAcceleration(1.0);
		// finger_pos_tasks[i]->_otg->setMaxLinearJerk(5.0);

		// finger_pos_tasks[i]->_otg->setMaxAngularVelocity(M_PI/1.5);
		// finger_pos_tasks[i]->_otg->setMaxAngularAcceleration(3*M_PI);
		// finger_pos_tasks[i]->_otg->setMaxAngularJerk(15*M_PI);

		finger_pos_tasks[i]->_kp = 50.0;
		finger_pos_tasks[i]->_kv = 17.0;
	}

	VectorXd one_finger_computed_torques =  VectorXd::Zero(robot_dof); 
	MatrixXd all_pos_task_torques = VectorXd::Zero(robot_dof); 

	// Vector3d x_init = finger_pos_task->_current_position;
	// Matrix3d R_init = finger_pos_task->_current_orientation;

	// if(!flag_simulation) {
	// 	force_bias << 1.50246,   -8.19902,  -0.695169,  -0.987652,   0.290632, -0.0453239;
	// 	tool_mass = 0.33;
	// 	tool_com = Vector3d(-0.00492734, -0.00295005,   0.0859595);
	// }

	// setup redis keys to be updated with the callback
	redis_client.createReadCallback(0);
	redis_client.createWriteCallback(0);

	// Objects to read from redis
    redis_client.addEigenToReadCallback(0, JOINT_ANGLES_KEY, robot->_q);
    redis_client.addEigenToReadCallback(0, JOINT_VELOCITIES_KEY, robot->_dq);
	// redis_client.addEigenToReadCallback(0, JOINT_ANGLES_DESIRED_KEY, joint_task->_desired_position);
	
	// Objects to write to redis
	redis_client.addEigenToWriteCallback(0, ROBOT_COMMAND_TORQUES_KEY, command_torques);

	// setup data logging
	string folder = "../../0-allegro_hand_grasping/data_logging/data/";
	string filename = "data";
    auto logger = new Logging::Logger(100000, folder + filename);
	
	// Vector3d log_robot_ee_position = x_init;
	// Vector3d log_robot_ee_velocity = Vector3d::Zero();
	VectorXd log_joint_angles = robot->_q;
	VectorXd log_joint_velocities = robot->_dq;
	VectorXd log_joint_command_torques = command_torques;
    // Vector3d log_desired_force = Vector3d::Zero();

	// logger->addVectorToLog(&log_robot_ee_position, "robot_ee_position");
	// logger->addVectorToLog(&log_robot_ee_velocity, "robot_ee_velocity");
	logger->addVectorToLog(&log_joint_angles, "joint_angles");
	logger->addVectorToLog(&log_joint_velocities, "joint_velocities");
	logger->addVectorToLog(&log_joint_command_torques, "joint_command_torques");
    // logger->addVectorToLog(&log_desired_force, "desired_force");

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

		// read robot state
		redis_client.executeReadCallback(0);

		if(flag_simulation) {
			robot->updateModel();
			// robot->coriolisForce(robot_coriolis);
		}
		else {
			robot->updateKinematics();
			// robot->_M = mass_from_robot;
			robot->updateInverseInertia();
			// coriolis = coriolis_from_robot;
		}

		// Set Task Hirearchy
		N_prec.setIdentity(robot_dof,robot_dof);

		finger_task_Jacobian = MatrixXd::Zero(3,robot_dof);
		combined_task_Jacobian = MatrixXd::Zero(int(3*4),robot_dof);
		for (int i = 0; i < 4; i++) { 		// Construct combined Jacobian for all finger tasks 
			finger_pos_tasks[i]->updateTaskModel(N_prec);
			robot->Jv(finger_task_Jacobian, hand_ee_link_names[i], hand_ee_pos_in_link);
			// cout<<"finger task Jacobian " + to_string(i) << endl << finger_task_Jacobian  << endl;
			combined_task_Jacobian.block(i*3, 0, 3, robot_dof) = finger_task_Jacobian;
			// cout<<"combined task Jacobian after " + to_string(i) + " fingers added" << endl << combined_task_Jacobian  << endl;
		}
		// cout<<"combined task Jacobian" << endl << combined_task_Jacobian  << endl;
		robot->nullspaceMatrix(N_prec, combined_task_Jacobian);
		// cout<<"nullspace Matrix" << endl << N_prec  << endl;

		joint_task->updateTaskModel(N_prec);
		// joint_task->updateTaskModel(MatrixXd::Identity(robot_dof,robot_dof));

		if(state == INIT) {
			robot->gravityVector(g);

			joint_task->computeTorques(joint_task_torques);
			command_torques = joint_task_torques + robot_coriolis;
			// command_torques = joint_task_torques + coriolis + g;

			// if((joint_task->_desired_position - joint_task->_current_position).norm() < 0.2) {
			// 	// Reinitialize controllers
			// 	for (int i = 0; i < 4; i++) {
			// 		finger_pos_tasks[i]->reInitializeTask();
			// 	}
			// 	joint_task->reInitializeTask();

			// 	joint_task->_kp = 50.0;
			// 	joint_task->_kv = 13.0;
			// 	joint_task->_ki = 0.0;

			// 	state = CONTROL;
			// }
			// joint_task->_desired_position(0) += M_PI / 3;
			// joint_task->_desired_position(4) -= M_PI / 3;
			// joint_task->_desired_position(8) += M_PI / 3;
			// joint_task->_desired_position(12) -= M_PI / 3;

			finger_pos_task_0->_desired_position <<  0.50000, 0.0759791, 0.7000;
			finger_pos_task_1->_desired_position <<  0.50000, 0.000, 0.7000;
			finger_pos_task_2->_desired_position <<  0.50000, -0.0561763, 0.7000;
			finger_pos_task_3->_desired_position <<  0.50000, 0.103486, 0.57549;


			state = CONTROL;
		}

		else if(state == CONTROL) {

			all_pos_task_torques = VectorXd::Zero(robot_dof); 
			// try	{
				for (int i = 0; i < 4; i++) {
					finger_pos_tasks[i]->computeTorques(one_finger_computed_torques);
					all_pos_task_torques += one_finger_computed_torques; // each pos task generates torques for all joints, with only the relevant finger joints being nonzero
				}
			// }
			// catch(exception e) {
			// 	cout << "control cycle: " << controller_counter << endl;
			// 	cout << "error in the torque computation of finger_pos_task:" << endl;
			// 	cerr << e.what() << endl;
			// 	cout << "setting torques to zero for this control cycle" << endl;
			// 	cout << endl;
			// 	// finger_pos_task_torques.setZero(); // set task torques to zero, TODO: test this
			// }
			joint_task->_desired_position = robot->_q; // use current robot joint positions as desired position
			joint_task->computeTorques(joint_task_torques);

			command_torques = all_pos_task_torques;
			command_torques = all_pos_task_torques + joint_task_torques;
			// command_torques = joint_task_torques + robot_coriolis;
			// command_torques = joint_task_torques;
			// command_torques = 10*VectorXd::Random(robot_dof);
			// command_torques = VectorXd::Zero(robot_dof);
			// cout<<"Control torques: " << endl << command_torques  << endl;
			
		}

		// if (activeStateString == "PREGRASP")
		// 	q_des_hand << 0.11, 0.34, 1.3, 1, 0.05, 0.43, 1.1, 1.3, -0.051, 0.46, 0.98, 1.2, 1.4, 0.075, 0.04, 1.6;
		// if (activeStateString == "TOUCH")
		// 	q_des_hand << 0.12, 0.53, 1.4, 0.88, 0.15, 0.67, 1.1, 1.1, -0.047, 0.46, 0.98, 1.2, 1.4, 0.076, 0.24, 1.3;
		// if (activeStateString == "GRASP")
		// 	q_des_hand << 0.062, 0.52, 1.4, 0.88, 0.21, 0.66, 1.1, 1.1, -0.052, 0.46, 0.98, 1.2,  1.4, 0.1, 0.45, 1.2;

		// if (activeStateString == "PREGRASP")
		// 	q_des_hand << 0.11, 0.34, 1.3, 1, 0.05, 0.43, 1.1, 1.3, -0.051, 0.46, 0.98, 1.2, 0, 1.2, 0.8, 1.6;
		// else if (activeStateString == "TOUCH")
		// 	q_des_hand << 0.12, 0.53, 1.4, 0.88, 0.15, 0.67, 1.1, 1.1, -0.047, 0.46, 0.98, 1.2, 0, 1.2, 0.8, 1.3;
		// else if (activeStateString == "GRASP")
		// 	q_des_hand << 0.062, 0.52, 1.4, 0.88, 0.21, 0.66, 1.1, 1.1, -0.052, 0.46, 0.98, 1.2,  0, 1.3, 0.8, 1.2;
		// else
		// 	q_des_hand << 0.11, 0.34, 1.3, 1, 0.05, 0.43, 1.1, 1.3, -0.051, 0.46, 0.98, 1.2, 0, 1.2, 0.8, 1.6;

		// write control torques
		redis_client.executeWriteCallback(0);

		// update logger values
		// Vector3d ee_pos = Vector3d::Zero();
		// Vector3d ee_vel = Vector3d::Zero();
		// robot->position(ee_pos, robot_ee_link_name, robot_ee_pos_in_link);
		// robot->linearVelocity(ee_vel, robot_ee_link_name, robot_ee_pos_in_link);
		
		// log_robot_ee_position = ee_pos;
		// log_robot_ee_velocity = ee_vel;
		log_joint_angles = robot->_q;
		log_joint_velocities = robot->_dq;
		log_joint_command_torques = command_torques;
        // log_desired_force = finger_pos_task->_desired_force;
	
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
