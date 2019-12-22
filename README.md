# Tank-Level-Control-with-Reinforcment-Learning

*This repository is still being updated for detailed explaiation of the system and a batter navigation through the code. All the codes have been already uploaded.*

This repository demonstrates the Reinforcement Learning **TD Q-Learning algorithm** to control the level of the tank.


# System Information
<img src="Assets/system.png" height="70%" width="70%" >
<img src="Assets/material_balance.png" height="80%" width="80%">
<!---The system, as shown in the figure, has one inlet and outlet. The outlet flow form the tank depends on the level of   the liquid in the tank.--> 

# State - Action
<img src="Assets/state_action.png" height="30%" width="30%" >
In order to track multiple setpoints, the state for this system is the current setpoint tracking error ( difference between setpoint and height ). Simillarly, the action is the necessary change in the current input.

# An Example
<img src="Assets/Example.png" height="70%" width="70%" >
