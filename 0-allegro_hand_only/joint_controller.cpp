// This example tests the haptic device driver and the open-loop bilateral teleoperation controller.

#include "Sai2Model.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "tasks/JointTask.h"
#include "tasks/PositionTask.h"
#include "tasks/PosOriTask.h"
#include "../src/Logger.h"

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
const string fingertip_link_names[] = {"link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"};
const Vector3d fingertip_pos_in_link = Vector3d(0.0, 0.0, 0.035);
Vector3d finger_tip_pos = Vector3d::Zero();

// define upper and lower joint limits
Eigen::VectorXd finger_joint_angle_upper = Eigen::VectorXd::Zero(4);
Eigen::VectorXd finger_joint_angle_lower  = Eigen::VectorXd::Zero(4);
Eigen::VectorXd thumb_joint_angle_upper  = Eigen::VectorXd::Zero(4);
Eigen::VectorXd thumb_joint_angle_lower  = Eigen::VectorXd::Zero(4);


MatrixXd GenerateJointGrid(const VectorXd& lower_bound, const VectorXd& upper_bound){
    const int NUM_JOINTS = 4;
    const int NUM_POINTS_PER_JOINT = 10;
    MatrixXd joint_grid = MatrixXd::Zero(pow(NUM_POINTS_PER_JOINT, NUM_JOINTS), NUM_JOINTS);
    VectorXd steps = VectorXd::Zero(NUM_JOINTS);
    for (int joint_id = 0; joint_id < NUM_JOINTS; joint_id++){
        steps[joint_id] = (upper_bound[joint_id] - lower_bound[joint_id]) / NUM_POINTS_PER_JOINT;
    }

    for(int i=0; i<NUM_POINTS_PER_JOINT; i++){
        for(int j=0; j<NUM_POINTS_PER_JOINT; j++){
            for(int k=0; k<NUM_POINTS_PER_JOINT; k++){
                for(int l=0; l<NUM_POINTS_PER_JOINT; l++){
                    VectorXd delta = VectorXd::Zero(4);
                    delta << i * steps[0], j * steps[1], k * steps[2], l * steps[3];
                    joint_grid.row(i * pow(NUM_POINTS_PER_JOINT, 3) + j * pow(NUM_POINTS_PER_JOINT, 2) + k * NUM_POINTS_PER_JOINT + l) = lower_bound + delta;
                }
            }
        }
    }

    return joint_grid;
}



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
	T_world_robot.translation() = Vector3d(0.35, 0.0, 0.6);
	auto robot = new Sai2Model::Sai2Model(robot_file, false, T_world_robot);

    // upper and lower limits
    finger_joint_angle_upper << 0.57181227113054078, 1.7367399715833842, 1.8098808147084331, 1.71854352396125431;
    finger_joint_angle_lower << -0.59471316618668479, -0.29691276729768068, -0.27401187224153672, -0.32753605719833834;
    thumb_joint_angle_upper << -1.3968131524486665, 1.1630997544532125, 1.6440185506322363, 1.7199110516903878;
    thumb_joint_angle_lower << -0.2635738998060688, -0.10504289759570773, -0.18972295140796106, -0.16220637207693537;


	robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
	robot->updateModel();
    cout << "joint position: " << robot->_q << endl;
	cout <<"joint velocity: " << robot->_dq << endl;

	int robot_dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(robot_dof);
	VectorXd robot_coriolis = VectorXd::Zero(robot_dof);
    VectorXd gravity = VectorXd::Zero(robot_dof);
	MatrixXd N_prec = MatrixXd::Identity(robot_dof,robot_dof);
	MatrixXd finger_task_Jacobian = MatrixXd::Zero(3,robot_dof);
	MatrixXd combined_task_Jacobian = MatrixXd::Zero(int(3*4),robot_dof);


	// controller_state
	int state = INIT;

	// joint task
	auto joint_task = new Sai2Primitives::JointTask(robot);
	joint_task->_use_interpolation_flag = false;
	joint_task->_use_velocity_saturation_flag = true;
	joint_task->setDynamicDecouplingFull();

	VectorXd joint_task_torques = VectorXd::Zero(robot_dof);
	joint_task->_kp = 50.0;
	joint_task->_kv = 5.0;
	joint_task->_ki = 0.0;

	Eigen::VectorXd g(robot_dof); //joint space gravity vector
    joint_task->_desired_position = robot->_q; // use current robot config as init config
	redis_client.setEigenMatrixJSON(JOINT_ANGLES_DESIRED_KEY, robot->_q);

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
	string folder = "";
	string filename = "kinematic_characterization_finger_4_1";
    auto logger = new Logging::Logger(10000, folder + filename);
	

	VectorXd log_finger_tip = finger_tip_pos;
	logger->addVectorToLog(&log_finger_tip, "log_finger_tip");

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

    MatrixXd joint_grids = GenerateJointGrid(thumb_joint_angle_lower, thumb_joint_angle_upper);
    int current_config_idx = 0;
    int max_configurations = joint_grids.rows();
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
		joint_task->updateTaskModel(N_prec);

		if(state == INIT) {
			if (current_time - prev_time < 2.25){
                joint_task->_desired_position = robot->_q;
                joint_task->_desired_position.segment(12, 4) = thumb_joint_angle_upper;
                joint_task->computeTorques(joint_task_torques);
			    command_torques = joint_task_torques;
            } else {
                state = CONTROL;
                cout << "joint position: " << robot->_q << endl;
	            cout <<"joint velocity: " << robot->_dq << endl;
            }
		}

		else if(state == CONTROL) {
            if (current_time - prev_time > 0.25){
                robot->position(finger_tip_pos, "link_15.0_tip", fingertip_pos_in_link);
                log_finger_tip = finger_tip_pos;
                logger->log();
                if (current_config_idx < max_configurations){
                    joint_task->_desired_position.segment(12, 4) = joint_grids.row(current_config_idx);
                    current_config_idx++;
                }
                prev_time = current_time;
            }
			joint_task->computeTorques(joint_task_torques);
			command_torques = joint_task_torques;		
		}

		// write control torques
		redis_client.executeWriteCallback(0);	
		controller_counter++;
	}

	// stop logger
	logger->stop_log();

	//// Send zero force/torque to robot ////
	redis_client.setEigenMatrixJSON(ROBOT_COMMAND_TORQUES_KEY, command_torques);


	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
	std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}
