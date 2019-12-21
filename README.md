# Tank-Level-Control-with-Reinforcment-Learning

*This repository is still being updated for a batter navigation through the code. All the codes have been already uploaded.

This repository demonstrates the reinforcement learning TD Q-Learning algorithm to control the level of the tank


# System Information
![](Assets/system.png) <!-- .element height="50%" width="50%" -->

The system, as shown in the figure, has one inlet and outlet. The outlet flow form the tank depends on the level of the liquid in the tank. 

# State - Action
![](Assets/state_action.png)

In order to track multiple setpoints, the state for this system is the current setpoint tracking error ( difference between setpoint and height ). Simillarly the action is the necessary change in the current input.

# An Example
![](Assets/Example.png)
