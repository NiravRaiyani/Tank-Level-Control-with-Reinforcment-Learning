import numpy as np



class TankLevel:
    def __init__(self, set_point, initial_state, initial_action):
        '''

        set_point: Set point of the system
        initial_state: Initial state of the system
        initial_action: Initial Input to the system
        '''
        self.Sp = set_point
        self.S0 = initial_state
        self.A0 = initial_action
        self.state_table = list(np.round(np.linspace(-1.2, 1.2, 25), 2))
        self.action_table = list(np.round(np.linspace(-0.014, 0.014, 29), 3))

    def RL_state(self, s):
        '''
        This  function converts the real system state to RL environment state ( An integer representing the state of the system ).
        s: Real of the system

        '''
        # RL_state
        state = np.round(s - self.Sp, 1)

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
        self.A0 = self.get_action(a)

        # Finding the steady state of the system corresponding to the action 'A'
        self.S0 = self.next_state(self.A0)

        reward = self.reward_calc(self.S0)

        next_s = self.convert_state(self.S0)

        if self.state_table[next_s] == 0:
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
        A = min(0.014, A)
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
        S = np.round(S - self.Sp, 1)
        if S == 0:
            reward = 100
        else:
            reward = -np.square(S)*100
        return reward

    def convert_state(self, S):
        '''
        Converting the actual state to RL state
        S: Actual state of the system
        sp: Set point
        return: RL environment state of the system
        '''

        # Converting the actual state to RL state
        state = np.round(S - self.Sp, 1)

        s = self.state_table.index(state)

        return s

    def test_step(self, a):
        '''
        Calculating the response of the system for the given input
        a:
        :return:
        '''
        # Getting the Actual action
        self.A0 = self.get_action(a)
        Ts = 0.1
        # RL State

        # Getting the Actual State
        for i in range(10):
            self.S0 = self.S0 + (Ts / 0.79) * (self.A0 - 0.0133 * np.sqrt(self.S0))

            # To avoid NaN s
            self.S0 = np.round(self.S0, 6)
        next_s = self.convert_state(self.S0)
        reward = self.reward_calc(self.S0)
        if self.state_table[next_s] == 0:
            done = True
        else:
            done = False


        return next_s, reward, done


    def diff(self, S0, A):

        # Getting the Actual State
        Ts = 0.1
        for i in range(10):
            S0 = self.S0 + (Ts / 0.79) * (A - 0.0133 * np.sqrt(S0))

        s = self.convert_state(S0)
        return s