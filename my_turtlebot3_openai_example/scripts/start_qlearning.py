#!/usr/bin/env python

import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.turtlebot3 import turtlebot3_world


if __name__ == '__main__':

    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment 사용할 task_env 의 id와 맞춰줌. openai_ros.task_envs.turtlebot3 import turtlebot3_world 여기에 정의 되어있음.
    env = gym.make('TurtleBot3World-v0')
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    outdir = pkg_path + '/training_results'
    #기존 환경의 기능을 확장
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")
    #time을 저장하고자 하는 배열을 만듬
    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    # 각각  파라미터 의미  필요  
    Alpha = rospy.get_param("/turtlebot3/alpha")
    Epsilon = rospy.get_param("/turtlebot3/epsilon")
    Gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    nsteps = rospy.get_param("/turtlebot3/nsteps")

    running_step = rospy.get_param("/turtlebot3/running_step")

    # Initialises the algorithm that we are going to use for learning
    #qlearn 알고리즘 대략 설명 필요 
    #qlearn 알고리즘에 위에서 config file에서 읽어온 alpha , gamma, epsilon 값을 setting 해준다. 
    #그리고 Bot이 취하는 행동이 몇가지인지 알려준다. 여기서는 config 파일에 
    #n_actions: 3 # We have 3 actions, Forwards,TurnLeft,TurnRight 으로 정의 되어있고 
    #task_env(openai_ros.task_envs.turtlebot3 import turtlebot3_world)파일이 이값을 읽어와서 자신의 action_space 변수에 저장, 
    #이를 여기서 읽어와서 Qlearn에 넣어줌. 
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    #initial_epsilon 값을 위에 샛팅한 qlearning 알고리즘으로 부터 받아옴. epsilon=0.05 고정값 사용  
    #qlearn 에서 epsilon 이란

    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.logdebug("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logdebug(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                rospy.logdebug("NOT DONE")
                state = nextState
            else:
                rospy.logdebug("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
