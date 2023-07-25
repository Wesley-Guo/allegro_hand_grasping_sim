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
string JOINT_ANGLES_SIM_KEY = "sai2::AllegroGraspSim::0::simviz::sensors::q";
string JOINT_VELOCITIES_SIM_KEY = "sai2::AllegroGraspSim::0::simviz::sensors::dq";
string JOINT_ANGLES_DESIRED_SIM_KEY = "sai2::AllegroGraspSim::0::simviz::control::q_des";
string ROBOT_COMMAND_TORQUES_SIM_KEY = "sai2::AllegroGraspSim::0::simviz::actuators::tau_cmd";

string CONTROL_MODE = "allegroHand::controller::control_mode";
string JOINT_ANGLES_KEY = "allegroHand::sensors::joint_positions";
string JOINT_VELOCITIES_KEY = "allegroHand::sensors::joint_velocities";
string JOINT_ANGLES_DESIRED_KEY = "allegroHand::controller::joint_positions_commanded";
string ROBOT_COMMAND_TORQUES_KEY = "allegroHand::controller::joint_torques_commanded";

RedisClient redis_client;


const double control_loop_freq = 1000.0;

// set control link and point for posori tasks
const string fingertip_link_names[] = {"link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip"};
const Vector3d fingertip_pos_in_link = Vector3d(0.0,0.0,0.035);

// const bool flag_simulation = false;
const bool flag_simulation = false;

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
	T_world_robot.translation() = Vector3d(0.35, 0.0, 0.6);
	auto robot = new Sai2Model::Sai2Model(robot_file, false, T_world_robot);
	int robot_dof = robot->dof();

	if (flag_simulation){
		robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
		robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);
	} else {
		robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
		robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
	}
	robot->updateModel();

	VectorXd command_torques = VectorXd::Zero(robot_dof);
	VectorXd robot_coriolis = VectorXd::Zero(robot_dof);
	MatrixXd N_prec = MatrixXd::Identity(robot_dof,robot_dof);
    MatrixXd N_task = MatrixXd::Identity(robot_dof,robot_dof);
	MatrixXd finger_task_Jacobian = MatrixXd::Zero(3,robot_dof);
	MatrixXd combined_task_Jacobian = MatrixXd::Zero(3 * 4,robot_dof);
    VectorXd q_mid = VectorXd::Zero(robot_dof);
    q_mid << robot->_q;

	// controller_state
	int state = INIT;

	// joint task for initialization
	auto joint_task = new Sai2Primitives::JointTask(robot);
	joint_task->_use_interpolation_flag = false;
	joint_task->_use_velocity_saturation_flag = true;
	joint_task->setDynamicDecouplingNone();

	VectorXd joint_task_torques = VectorXd::Zero(robot_dof);
	joint_task->_kp = 50.0;
	joint_task->_kv = 14.0;
	joint_task->_ki = 0.0;

	double kpj = 0.0;
	if (flag_simulation){
		kpj = 50.0;	
	} else {
		kpj = 5.0;
	}
    double kvj = 25.0;

	Eigen::VectorXd g(robot_dof); //joint space gravity vector
    joint_task->_desired_position = robot->_q; // use current robot config as init config
	if (flag_simulation) {
		redis_client.setEigenMatrixJSON(JOINT_ANGLES_DESIRED_SIM_KEY, robot->_q);
	} else {
		redis_client.setEigenMatrixJSON(JOINT_ANGLES_DESIRED_KEY, robot->_q);
	}
	

	// define tasks for each fingertip
    // TODO: make this shapred_ptr
	double POSITION_GAIN = 0.0;
	if (flag_simulation){
		POSITION_GAIN = 50.0;
	} else {
		POSITION_GAIN = 7500.0;
	}
	auto finger_pos_task_0 = new Sai2Primitives::PositionTask(robot, fingertip_link_names[0], fingertip_pos_in_link);
	finger_pos_task_0->_use_velocity_saturation_flag = false;
	finger_pos_task_0->_saturation_velocity = 1.0;
	finger_pos_task_0->_kp = POSITION_GAIN;
	finger_pos_task_0->_kv = 15.0;
    auto finger_pos_task_1 = new Sai2Primitives::PositionTask(robot, fingertip_link_names[1], fingertip_pos_in_link);
	finger_pos_task_1->_use_velocity_saturation_flag = false;
	finger_pos_task_1->_saturation_velocity = 1.0;
	finger_pos_task_1->_kp = POSITION_GAIN;
	finger_pos_task_1->_kv = 15.0;
    auto finger_pos_task_2 = new Sai2Primitives::PositionTask(robot, fingertip_link_names[2], fingertip_pos_in_link);
	finger_pos_task_2->_use_velocity_saturation_flag = false;
	finger_pos_task_2->_saturation_velocity = 1.0;
	finger_pos_task_2->_kp = POSITION_GAIN;
	finger_pos_task_2->_kv = 15.0;
    auto finger_pos_task_3 = new Sai2Primitives::PositionTask(robot, fingertip_link_names[3], fingertip_pos_in_link);
	finger_pos_task_3->_use_velocity_saturation_flag = false;
	finger_pos_task_3->_saturation_velocity = 1.0;
	finger_pos_task_3->_kp = POSITION_GAIN;
	finger_pos_task_3->_kv = 15.0;

	std::vector<Sai2Primitives::PositionTask*> finger_pos_tasks;
    finger_pos_tasks.push_back(finger_pos_task_0);
    finger_pos_tasks.push_back(finger_pos_task_1);
    finger_pos_tasks.push_back(finger_pos_task_2);
    finger_pos_tasks.push_back(finger_pos_task_3);

	std::vector<Eigen::Vector3d> finger_target_positions;
	Eigen::Vector3d finger_position_desired = Eigen::Vector3d::Zero();
	finger_position_desired << 0.08, 0.05, -0.02;
	finger_target_positions.push_back(finger_position_desired);
	finger_position_desired << 0.08, 0.0, -0.02;
	finger_target_positions.push_back(finger_position_desired);
	finger_position_desired << 0.08, -0.04, -0.02;
	finger_target_positions.push_back(finger_position_desired);
	finger_position_desired << 0.08, 0.09, 0.02;
	finger_target_positions.push_back(finger_position_desired);

	std::vector<Eigen::Vector3d> finger_current_positions;
	Eigen::Vector3d finger_current_position = Eigen::Vector3d::Zero();
	robot->position(finger_current_position, fingertip_link_names[0], fingertip_pos_in_link);
	finger_current_positions.push_back(finger_current_position);
	robot->position(finger_current_position, fingertip_link_names[1], fingertip_pos_in_link);
	finger_current_positions.push_back(finger_current_position);
	robot->position(finger_current_position, fingertip_link_names[2], fingertip_pos_in_link);
	finger_current_positions.push_back(finger_current_position);
	robot->position(finger_current_position, fingertip_link_names[3], fingertip_pos_in_link);
	finger_current_positions.push_back(finger_current_position);

	int current_finger_idx = 0;

	Affine3d T_world_finger_0 = Affine3d::Identity();

	VectorXd one_finger_computed_torques =  VectorXd::Zero(robot_dof); 
	VectorXd all_pos_task_torques = VectorXd::Zero(robot_dof); 

	// setup redis keys to be updated with the callback
	redis_client.createReadCallback(0);
	redis_client.createWriteCallback(0);

	
	if (flag_simulation){
		// Objects to read from redis
		redis_client.addEigenToReadCallback(0, JOINT_ANGLES_SIM_KEY, robot->_q);
		redis_client.addEigenToReadCallback(0, JOINT_VELOCITIES_SIM_KEY, robot->_dq);
		
		// Objects to write to redis
		redis_client.addEigenToWriteCallback(0, ROBOT_COMMAND_TORQUES_SIM_KEY, command_torques);
	} else {
		// Objects to read from redis
		redis_client.addEigenToReadCallback(0, JOINT_ANGLES_KEY, robot->_q);
		redis_client.addEigenToReadCallback(0, JOINT_VELOCITIES_KEY, robot->_dq);
		
		// Objects to write to redis
		redis_client.addEigenToWriteCallback(0, ROBOT_COMMAND_TORQUES_KEY, command_torques);
	}

	// setup data logging
	string folder = "";
	string filename = "data";
    auto logger = new Logging::Logger(10000, folder + filename);
	
    Vector3d finger_tip_pos  = Vector3d::Zero();
    Vector3d finger_tip_vel  = Vector3d::Zero();
    robot->position(finger_tip_pos, fingertip_link_names[0], fingertip_pos_in_link);
    std::cout << "current position: " << finger_tip_pos << std::endl;

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
	double prev_time = 0;
	// double dt = 0;
	bool fTimerDidSleep = true;
	double start_time = timer.elapsedTime(); //secs
	prev_time = start_time;

	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		current_time = timer.elapsedTime() - start_time;

		// read robot state
		redis_client.executeReadCallback(0);

		robot->updateModel();

		// Set Task Hirearchy
		N_prec.setIdentity(robot_dof,robot_dof);
        N_task.setIdentity(robot_dof,robot_dof);

		if(state == INIT) {
            joint_task->updateTaskModel(N_prec);

			joint_task->computeTorques(joint_task_torques);
			command_torques.setZero();
			command_torques = joint_task_torques;

			state = CONTROL;
		}

		else if(state == CONTROL) {
			all_pos_task_torques = VectorXd::Zero(robot_dof); 

			if (current_time - prev_time > 3.25){
				current_finger_idx += 1;
				current_finger_idx = current_finger_idx % 4;
				prev_time = current_time;
			}

			for (int i = 0; i < 4; i++){
				finger_pos_tasks[i]->updateTaskModel(N_prec);
				if (current_finger_idx == i){
					finger_pos_tasks[i]->_desired_position << finger_target_positions[i];
				} else {
					finger_pos_tasks[i]->_desired_position << finger_current_positions[i];
				}
            	
				robot->Jv(finger_task_Jacobian, fingertip_link_names[i], fingertip_pos_in_link);
				combined_task_Jacobian.block(i*3, 0, 3, robot_dof) = finger_task_Jacobian;
			}

		    robot->nullspaceMatrix(N_task, combined_task_Jacobian);

			try	{
				for (int i = 0; i < 4; i++) {
					finger_pos_tasks[i]->computeTorques(one_finger_computed_torques);
					all_pos_task_torques += one_finger_computed_torques; // each pos task generates torques for all joints, with only the relevant finger joints being nonzero
					// cout<<"one finger task torque" << endl << one_finger_computed_torques << endl;
				}
			}
			catch(exception e) {
				cout << "control cycle: " << controller_counter << endl;
				cout << "error in the torque computation of finger_pos_task:" << endl;
				cerr << e.what() << endl;
				cout << "setting torques to zero for this control cycle" << endl;
				cout << endl;
			}
			command_torques = all_pos_task_torques + N_task * robot->_M *  (-kpj * (robot->_q - q_mid) - kvj * robot->_dq);			
            cout<<"commanded operational torques: " << endl << all_pos_task_torques << endl;
			cout<<"commanded posture torques: " << endl << N_task * (robot->_M * (-kpj * (robot->_q - q_mid) - kvj * robot->_dq)) << endl;
		}
		// write control torques
		redis_client.executeWriteCallback(0);

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
	if (flag_simulation){
		redis_client.setEigenMatrixJSON(ROBOT_COMMAND_TORQUES_SIM_KEY, command_torques);	
	} else {
		redis_client.setEigenMatrixJSON(ROBOT_COMMAND_TORQUES_KEY, command_torques);	
	}
	


	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
	std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}
