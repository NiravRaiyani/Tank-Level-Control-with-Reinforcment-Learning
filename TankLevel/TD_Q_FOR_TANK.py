import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from TankLevel_2 import *


############################### General Functions #######################################

def init_Q(s, a, type="zeros"):
    '''
    This function initializes the table of Action-value function for each state and action.
    :param s: No. of states
    :param a: NO. of possible action available
    :param type: "zeros", "Ones", "Random"
    :return: s x a dimensional matrix for action value function Q(s, a).
    '''
    if type == "ones":
        q = np.ones((s, a))

    if type == "zeros":
        q = np.zeros((s, a))

    if type == "random":
        q = np.random.random((s, a))

    return q


def e_greedy(no_a, e, q):
    """
    This function performs the epsilon greedy action selection
    :param no_a: No. of actions available
    :param e: Exploration parameter
    :param q: Action value function for the current state
    :return: epsilon greedy action
    """
    k = np.random.rand()
    if k < e:
        a = np.random.randint(0, no_a)
    else:
        a = np.argmax(q)
    return a




####################### Q -Learning #########################################
def TD_Q(alpha, gamma, epsilon, episodes, max_step):
    '''
    Training of the level controller in the RL - Environment by TD-Q learning

    '''

    # System Initial State
    S = 1.0

    # Set Point
    sp = 0.5

    # Initial Input to the system
    A = 0.0133

    # Initializing TankLevel Environment
    env = TankLevel(sp, S, A)

    # Defining the size of the state-space
    n_s, n_a = 25, 29

    # Initializing the Q-Table
    Q = init_Q(n_s, n_a, "zeros")

    # Variables for recording the progress of the training
    episode_reward = []
    state = []
    set_point = []

    # Episode Loop
    for i in tqdm(range(episodes)):
        t = 0
        total_reward = 0

        # Changing the Set point every 1000 episode
        if i>0 and i % 1000 == 0:
            sp_range = list(np.round(np.linspace(0, 1.2, 13), 1))
            env.Sp = np.float(np.random.choice(sp_range, 1))

        # Changing the exploration parameter every 500 episode
        if i>0 and i% 500 == 0:
            epsilon = max(0.1, epsilon - 0.01)

        # RL Environment state
        s = env.RL_state(S)

        # e - greedy action selection
        a = e_greedy(n_a, epsilon, Q[s, :])

        # Step loop
        while t < max_step:

            # Taking a step in the environment
            next_s, reward, done = env.step(a)

            # Action selection for the next state according to behaviour policy(i.e. e - greedy)
            next_a = e_greedy(n_a, epsilon, Q[next_s, :])

            # finding the action with maximum return
            max_a = np.argmax(Q[next_s,:])
            total_reward += reward

            # Action-value function update
            if done:
                Q[s, a] += alpha*(reward  - Q[s, a])
            else:
                Q[s, a] += alpha*(reward + gamma*Q[next_s, max_a] - Q[s, a])
                # or
                # Q[s, a] += alpha*(reward + gamma*np.max(Q[next_s, :]) - Q[s, a])

            # Recording the state and set point
            state.append(env.S0)
            set_point.append(env.Sp)

            t += 1
            s, a = next_s, next_a

            if done:
                episode_reward.append(total_reward)
                #print("success in episode:{}", format(i))
                episode_reward.append(total_reward)

                #plt.plot(episode_reward)
                #plt.show()

    np.savetxt('Q_Mattrix', Q)       #episode_reward.append(total_reward)
    return episode_reward,  Q, state, set_point



def test(S, A, epsilon = 0) :
    '''
    Testing the level controller trained by TD-Q.
    '''

    # Initial Set Point
    sp = 0.9

    # Total number of available action
    n_a = 29

    # Loading the trained Q-Table
    Q = np.loadtxt('Q_Mattrix')

    # Initializing the TankLevel Environment
    env = TankLevel(sp, S, A)

    # Variables to record the progress
    episode_reward = []
    set_point = []
    state = []
    action = []

    # Episode Loop
    for i in tqdm(range(n_test)):
        t = 0
        total_reward = 0

        # Changing the Set Point at every episode
        if i>0 and i % 1 == 0:
            sp_range = list(np.round(np.linspace(0, 1.1, 13), 1))
            env.Sp = np.random.choice(sp_range, 1)


        # RL Environment state
        s = env.RL_state(S)

        # Step loop
        while t < max_steps:

            # Updating the input every 2 seconds based on the current state
            if t % 2 == 0:

                # Completely greedy action selection
                a = e_greedy(n_a, epsilon, Q[s, :])

            # Taking an action in the environment
            next_s, reward, done = env.test_step(a)


            s = next_s
            total_reward += reward
            t += 1

            # Recording Outputs:
            set_point.append(env.Sp)
            state.append(env.S0)
            action.append(env.A0)

            if done:
                episode_reward.append(total_reward)
                #print("success set point{} and current state = {}", format(env.Sp, env.S0))

    return episode_reward, state, set_point, action





if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.9
    epsilon = 0
    episode = 1000
    max_steps = 500
    n_test = 15
    #train, Q, S, Sp  = TD_Q(alpha, gamma, epsilon, episode, max_steps)


    test, S, Sp, a = test(0.46, 0.0088, epsilon)

    plt.subplot(2, 1, 1)
    plt.plot(S)
    plt.plot(Sp)
    plt.legend(['h', 'Set point'])
    plt.xlabel("Steps/Time(s)")
    plt.ylabel("Tank Level(m)")

    plt.subplot(2, 1, 2)
    plt.plot(a)
    plt.xlabel("Steps/Time(s)")
    plt.ylabel("Input (m^3/s)")
    plt.show()
    print("Done!!!")





