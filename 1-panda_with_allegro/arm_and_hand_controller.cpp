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

const string arm_robot_file = "./resources/panda_arm.urdf";
const string hand_robot_file = "./resources/allegro_hand.urdf";

// redis keys:
// - read:
const std::string JOINT_ANGLES_SIM_KEY = "sai2::panda_robot_with_allegro::sensors::q";
const std::string JOINT_VELOCITIES_SIM_KEY = "sai2::panda_robot_with_allegro::sensors::dq";

const std::string WRIST_POSITION_KEY = "mocap::right_hand::position";
const std::string WRIST_ORIENTATION_KEY = "mocap::right_hand::orientation";

// - write
const std::string ARM_TORQUES_COMMANDED_SIM_KEY = "sai2::panda_robot::actuators::tau";
const std::string HAND_TORQUES_COMMANDED_SIM_KEY = "sai2::allegro::actuators::tau";

const string CONTROLLER_RUNNING = "sai2::panda_robot::controller_running";

unsigned long long controller_counter = 0;

int main() {

	// start redis client
	auto redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

    const int hand_dof = 16;
	
    // Initialize Franka Arm Controller
	const string arm_link_name = "link7";
	const Vector3d arm_tcp_pos_in_link = Vector3d(0, 0, 0.1);

	// load robots
	auto arm_robot = new Sai2Model::Sai2Model(arm_robot_file, false);
	const int arm_dof = arm_robot->dof();
	VectorXd q = VectorXd::Zero(arm_dof + hand_dof);
	VectorXd dq = VectorXd::Zero(arm_dof + hand_dof);

	q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
	dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);
	arm_robot->_q = q.head(arm_dof);
	arm_robot->_dq = dq.head(arm_dof);

	VectorXd initial_q = arm_robot->_q;
	arm_robot->updateModel();

	Vector3d x_d = Vector3d::Zero();
	arm_robot->position(x_d, arm_link_name, arm_tcp_pos_in_link);

	Eigen::Matrix3d R_fixed =  Eigen::Matrix3d::Identity();
	R_fixed << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    arm_robot->rotation(R_fixed, arm_link_name);

	// prepare controller
	Eigen::VectorXd arm_posori_torques;
	Eigen::VectorXd arm_joint_torques;

	VectorXd command_torques = VectorXd::Zero(arm_dof);
	Eigen::MatrixXd N_prec = Eigen::MatrixXd::Identity(arm_dof, arm_dof);

	Sai2Primitives::PosOriTask *arm_posori_task = new Sai2Primitives::PosOriTask(arm_robot, arm_link_name, arm_tcp_pos_in_link);
	arm_posori_task->updateTaskModel(N_prec);

	Sai2Primitives::JointTask *arm_joint_task = new Sai2Primitives::JointTask(arm_robot);
	arm_joint_task->updateTaskModel(arm_posori_task->_N);

	arm_joint_task->_desired_position = arm_robot->_q;
	arm_joint_task->_desired_velocity.setZero(arm_robot->_dof);

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

		// read robot state from redis
		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_SIM_KEY);
		dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_SIM_KEY);
		arm_robot->_q = q.head(arm_dof);
		arm_robot->_dq = dq.head(arm_dof);
        arm_robot->updateModel();

		arm_posori_task->updateTaskModel(N_prec);
		arm_joint_task->updateTaskModel(arm_posori_task->_N);

		double t = timer.elapsedTime() - start_time;

		// Compute torques 
		arm_posori_task->_desired_orientation = R_fixed;
		arm_posori_task->_desired_angular_velocity = Eigen::Vector3d::Zero();

		arm_posori_task->_desired_velocity =  Eigen::Vector3d::Zero();
		x_d = redis_client.getEigenMatrixJSON(WRIST_POSITION_KEY);
		arm_posori_task->_desired_position = x_d;

		arm_posori_task->computeTorques(arm_posori_torques);
		arm_joint_task->computeTorques(arm_joint_torques);
		command_torques = arm_posori_torques + arm_joint_torques;

		// send to redis
		redis_client.setEigenMatrixJSON(ARM_TORQUES_COMMANDED_SIM_KEY, command_torques);

		controller_counter++;

	}

	command_torques.setZero();
	redis_client.setEigenMatrixJSON(ARM_TORQUES_COMMANDED_SIM_KEY, command_torques);
	redis_client.set(CONTROLLER_RUNNING, "0");

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
    std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";


	return 0;
}
