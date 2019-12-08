%% Calculating the response time of the system
clc;
clear all;
close all;

%% Initial Condition(Steady State Paramenter)
 h(1,1)= 0;
 F(1,1)= 0.010;
    
 % Sampling time
 Ts=0.1
 
 %% Running the process 
for i=1:1:10000
    
    % Step change in the input after 30 second 
    if i > 30
        F(1,1)=0.01456
    end
   
    h(i+1) = h(i)+((Ts/0.79)*(F(1,1) - 0.0133*sqrt(h(i))));
end

%% Plotting the response
plot(0:Ts:1000, h)
ylim([0, 1.3])    