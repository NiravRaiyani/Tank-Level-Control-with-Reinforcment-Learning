'''
Reinforcement Learning : Level in a tank system environment.

@ Author:   Nirav Raiyani
            University of Alberta
'''




import numpy as np
import matplotlib.pyplot as plt







def convert_action(A):

    assert -0.014 <= A <= 0.014
    # Preparing the Action table
    Action_table = list(np.round(np.linspace(-0.014, 0.014, 29), 3))

    # Fetching the value of the current state
    A = Action_table.index(A)

    return A

def next_state(a):

    s = np.divide(np.square(a), np.square(0.0133))
    return s

def reward_calc(S, sp):
    S = np.round(S - sp, 1)
    if S == 0:
        reward = 100
    else:
        reward = -np.square(S)
    return reward

def RL_state(s, sp):

    # RL_state
    state = np.round(s - sp, 1)

    # Preparing the Action table
    State_table = list(np.round(np.linspace(-1.2, 1.2, 25 ), 2))

    # Fetching the value of the current state
    a = State_table.index(state)

    return a

def step(A):

    '''
    Taking one step in to the environment
    param a: Action given by agent to be imposed on the input of the system
    param a0:
    return: next state, reward, done
    '''
    a0 = 0.002
    sp = 0.9
    a = max(0, get_action(A, a0))
    next_s = next_state(a)

    reward = reward_calc(next_s, sp)

    next_s = convert_state(next_s, sp)

    if next_s == 0:
        done = True
    else:
        done = False

    return next_s, reward, done


def get_action(A, a0):
    '''
    
    A: RL environment action (integer)
    return: Actual action (new input to the system)
    '''
    # Converting the RL_action to the input change delta u
    Action_table = list(np.round(np.linspace(-0.014, 0.014, 29), 3))

    # Converting the delta u to actual input to be used for system
    action = a0 + Action_table[A]
    # action = max(0, a0+Action_table[A])
    return action

def convert_state(s, sp):
    '''
    
    s: Actual state of the system 
    sp: Set point
    return: RL environment state of the system
    '''

    # Converting the actual state to RL_state
    state = np.round(s - sp, 1)

    # getting the integer value of the action for RL environment
    State_table = list(np.round(np.linspace(-1.2, 1.2, 25), 2))

    state = State_table.index(state)

    return state














































# class TankLevel:
#
#     def __init__(self, state, action, a_state, set_point):
#         self.state = state
#         self.action = action
#         self.original_state = a_state
#         self.set_point = set_point
#
#     def diff(self):
#         Ts = 0.1
#         s = np.zeros(10001)
#         self.s[0] = self.original_state
#         for t in range(0, 10000, 1):
#             self.s[t+1] = self.s[t] + (Ts/0.79)*(self.action - 0.0133*np.sqrt(self.s[t]))
#
#         return self.s
#
#     def Q_convert(self):
#         next_state = diff(self)
#
#
#
#
#     def reward(self):
#         if s==0:
#             reward = 100
#         else:
#             reward = -np.square(s)
#
#         return reward

#if __name__ == '__main__':
    #tank = TankLevel(0.1,0.0)
    #s = tank.diff(0.1, 0.010)
    #plt.plot(s)
    #plt.show()

class TankLevel():
    def __init__(self, set_point, initial_state, initial_action):
        '''

        set_point: Set point of the system
        initial_state: Initial state of the system
        initial_action: Initial Input to the system
        '''
        self.sp = set_point
        self.s0 = initial_state
        self.A0 = initial_action
        self.state_table = list(np.round(np.linspace(-1.2, 1.2, 25), 2))
        self.action_table = list(np.round(np.linspace(-0.014, 0.014, 29), 3))



    def RL_state(self, s):
        '''
        This  function converts the real system state to RL environment state ( An integer representing the state of the system ).
        s: Real of the system

        '''
        # RL_state
        state = np.round(s - self.sp, 1)

        # Fetching the value of the RL environment state
        S = self.state_table.index(state)

        return S


    def step(self, a):

        '''
        Taking one step in to the environment
        param a: Action given by agent to be imposed on the input of the system
        return: next state, reward, done
        done: True if the system reaches the set point, else False. Indicates the end of the episode.
        '''

        # Getting the Actual action
        A =  get_action(a)

        # Finding the steady state of the system corresponding to the action 'A'
        next_S = next_state(A)

        reward = reward_calc(next_S)

        next_s = convert_state(next_S)

        if self.state_table[next_s] == 0 :
            done = True
        else:
            done = False

        return next_s, reward, done


    def get_action(self, a):
        '''
        Converts RL environment action to Actual system action.
        a: RL environment action (integer)
        return: Actual action (new input to the system)
        '''

        # Converting the delta u to actual input to be used for system
        # max operation insures that actual input to the system remains non negative
        A = max(0, self.A0 + self.action_table[a])

        return A


    def next_state(self, A):
        '''
        calculates the corresponding steady state for the given A
        A: Actual input (or action) to the system
        return: Corresponding steady state of the system S
        '''
        S = np.divide(np.square(A), np.square(0.0133))
        return S


    def reward_calc(self, S):
        '''
        Calculates the reward for reaching the state S
        S: Current actual state of the system
        return: Reward
        '''
        S = np.round(S - self.sp, 1)
        if S == 0:
            reward = 1000
        else:
            reward = -np.square(S)*10
        return reward


    def convert_state(self, S):
        '''

        s: Actual state of the system
        sp: Set point
        return: RL environment state of the system
        '''

        # Converting the actual state to RL state
        state = np.round(S - self.sp, 1)

        s = self.state_table.index(state)

        return s