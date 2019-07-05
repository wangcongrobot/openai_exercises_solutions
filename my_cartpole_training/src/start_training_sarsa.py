#!/usr/bin/env python

import gym
import time
import numpy
import random
import time
import sarsa
from gym import wrappers

# ROS packages required
import rospy
import rospkg

# import our training environment
from openai_ros.task_envs.cartpole_stay_up import stay_up


if __name__ == '__main__':

    rospy.init_node('cartpole_gym', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('CartPoleStayUp-v0')
    rospy.loginfo ( "Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_cartpole_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/cartpole_v0/alpha")
    Epsilon = rospy.get_param("/cartpole_v0/epsilon")
    Gamma = rospy.get_param("/cartpole_v0/gamma")
    epsilon_discount = rospy.get_param("/cartpole_v0/epsilon_discount")
    nepisodes = rospy.get_param("/cartpole_v0/nepisodes")
    nsteps = rospy.get_param("/cartpole_v0/nsteps")
    running_step = rospy.get_param("/cartpole_v0/running_step")

    # Initialises the algorithm that we are going to use for learning
    sarsa = sarsa.Sarsa(actions=range(env.action_space.n),
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = sarsa.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE => " + str(x))

        cumulated_reward = 0
        done = False
        if sarsa.epsilon > 0.05:
            sarsa.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot

        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):

            rospy.loginfo("############### Start Step => "+str(i))
            # Pick an action based on the current state
            action = sarsa.chooseAction(state)
            rospy.loginfo ("Next action is: %d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            rospy.loginfo(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            #rospy.logwarn("############### State we were => " + str(state))
            #rospy.logwarn("############### Action that we took => " + str(action))
            #rospy.logwarn("############### Reward that action gave => " + str(reward))
            #rospy.logwarn("############### State in which we will start next step => " + str(nextState))
            sarsa.learn(state, action, reward, nextState, action)

            if not(done):
                state = nextState
            else:
                rospy.loginfo ("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.loginfo("############### END Step =>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            #rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logwarn ( ("EP: "+str(x+1)+" - [alpha: "+str(round(sarsa.alpha,2))+" - gamma: "+str(round(sarsa.gamma,2))+" - epsilon: "+str(round(sarsa.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))


    rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(sarsa.alpha)+"|"+str(sarsa.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()