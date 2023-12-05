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

const string robot_file = "./resources/panda_arm.urdf";
const string robot_name = "PANDA";

// redis keys:
// - read:
const std::string JOINT_ANGLES_KEY = "sai2::panda_robot_with_allegro::sensors::q";
const std::string JOINT_VELOCITIES_KEY = "sai2::panda_robot_with_allegro::sensors::dq";

const std::string WRIST_POSITION_KEY = "mocap::right_hand::position";
const std::string WRIST_ORIENTATION_KEY = "mocap::right_hand::orientation";
// - write
const std::string JOINT_TORQUES_COMMANDED_KEY = "sai2::panda_robot::actuators::tau";
const string ARM_CONTROL_RUNNING = "sai2::panda_robot::controller_running";

unsigned long long controller_counter = 0;

int main() {

	// start redis client
	auto redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	const string link_name = "link7";
	const Vector3d pos_in_link = Vector3d(0, 0, 0.1);

	// load robots
	auto robot = new Sai2Model::Sai2Model(robot_file, false);
	const int arm_dof = robot->dof();
	const int hand_dof = 16;
	VectorXd q = VectorXd::Zero(arm_dof + hand_dof);
	VectorXd dq = VectorXd::Zero(arm_dof + hand_dof);

	q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
	robot->_q = q.head(arm_dof);
	robot->_dq = dq.head(arm_dof);

	VectorXd initial_q = robot->_q;
	robot->updateModel();

	Vector3d x_d = Vector3d::Zero();
	robot->position(x_d, link_name, pos_in_link);

	Eigen::Matrix3d R_fixed =  Eigen::Matrix3d::Identity();
	R_fixed << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    robot->rotation(R_fixed, link_name);

	// prepare controller
	Eigen::VectorXd posori_torques;
	Eigen::VectorXd joint_torques;
	Eigen::VectorXd gravity_torques;
	VectorXd command_torques = VectorXd::Zero(arm_dof);
	Eigen::MatrixXd N_prec = Eigen::MatrixXd::Identity(arm_dof, arm_dof);

	Sai2Primitives::PosOriTask *posori_task = new Sai2Primitives::PosOriTask(robot, link_name, pos_in_link);
	posori_task->updateTaskModel(N_prec);

	Sai2Primitives::JointTask *joint_task = new Sai2Primitives::JointTask(robot);
	joint_task->updateTaskModel(posori_task->_N);

	joint_task->_desired_position = robot->_q;
	joint_task->_desired_velocity.setZero(robot->_dof);

	// create a timer
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(1000); 
	double start_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;

	redis_client.set(ARM_CONTROL_RUNNING, "1");
	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		double time = timer.elapsedTime() - start_time;

		// read robot state from redis
		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
		dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
		robot->_q = q.head(arm_dof);
		robot->_dq = dq.head(arm_dof);
        robot->updateModel();

		posori_task->updateTaskModel(N_prec);
		joint_task->updateTaskModel(posori_task->_N);

		double t = timer.elapsedTime() - start_time;

		// Compute torques 
		posori_task->_desired_orientation = R_fixed;
		posori_task->_desired_angular_velocity = Eigen::Vector3d::Zero();

		posori_task->_desired_velocity =  Eigen::Vector3d::Zero();
		x_d = redis_client.getEigenMatrixJSON(WRIST_POSITION_KEY);
		posori_task->_desired_position = x_d;

		posori_task->computeTorques(posori_torques);
		joint_task->computeTorques(joint_torques);
		command_torques = posori_torques + joint_torques;

		// send to redis
		redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, command_torques);

		controller_counter++;

	}

	command_torques.setZero();
	redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, command_torques);
	redis_client.set(ARM_CONTROL_RUNNING, "0");

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
    std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";


	return 0;
}
