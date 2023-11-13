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
const std::string JOINT_TORQUES_COMMANDED_KEY = "sai2::panda_robot_with_allegro::actuators::fgc";
const string CONTROLLER_RUNNING_KEY = "sai2::controller_running";

unsigned long long controller_counter = 0;

int main() {

	// start redis client
	auto redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load robots
	auto robot = new Sai2Model::Sai2Model(robot_file, false);
	int dof = robot->dof();
	VectorXd q = VectorXd::Zero(dof);
	VectorXd dq = VectorXd::Zero(dof);
	VectorXd tau = VectorXd::Zero(dof + 16);
	
	q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	robot->_q = q.head(dof);
	VectorXd initial_q = robot->_q;
	robot->updateModel();

	// prepare controller
	const string link_name = "link7";
	const Vector3d pos_in_link = Vector3d(0, 0, 0.1);
	VectorXd command_torques = VectorXd::Zero(dof);
	Eigen::MatrixXd N_prec = Eigen::MatrixXd::Identity(dof,dof);

	Sai2Primitives::RedundantArmMotion* motion_primitive = new Sai2Primitives::RedundantArmMotion(robot, link_name, pos_in_link);
	motion_primitive->enableGravComp();

	// create a timer
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(1000); 
	double start_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;

	redis_client.set(CONTROLLER_RUNNING_KEY, "1");
	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		double time = timer.elapsedTime() - start_time;

		// read robot state from redis
		q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
        dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);

		robot->_q = q.head(dof);
		robot->_dq = dq.head(dof);
		robot->updateModel();

		motion_primitive->updatePrimitiveModel(N_prec);

		double t = timer.elapsedTime() - start_time;
		Vector3d x_d = Vector3d::Zero();
		x_d = redis_client.getEigenMatrixJSON(WRIST_POSITION_KEY);

		Vector3d x = Vector3d::Zero(3);
		robot->position(x, link_name, pos_in_link);
		Eigen::Matrix3d R_fixed =  Eigen::Matrix3d::Identity();
		R_fixed << 1, 0, 0, 0, -1, 0, 0, 0, -1;

		// Compute torques 
		motion_primitive->_desired_orientation = R_fixed;
		motion_primitive->_desired_angular_velocity = Eigen::Vector3d::Zero();

		motion_primitive->_desired_velocity =  Eigen::Vector3d::Zero();
		motion_primitive->_desired_position = x_d;

		motion_primitive->computeTorques(command_torques);

		// send to redis
		tau = redis_client.getEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY);
		tau.head(dof) = command_torques;
		redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, tau);

		controller_counter++;

	}

	tau.setZero();
	redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, tau);
	redis_client.set(CONTROLLER_RUNNING_KEY, "0");

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
    std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";


	return 0;
}
