%% debug_jump.m
% ???????????????????????????????????????????????????????????????????????
%  Diagnostic script for 3-segment planar jump model
%  Run this to find working torque values before running animate_jump.m
%
%  Does two things:
%    1. Sweeps torque scales to find the right magnitude
%    2. Plots angles and COM velocity for the best scale
% ???????????????????????????????????????????????????????????????????????
clear; clc;

%% ?? Model Parameters ?????????????????????????????????????????????????
body_mass   = 70;
body_height = 1.75;

params.g    = 9.81;
params.m1   = 0.0145 * body_mass;
params.m2   = 0.0465 * body_mass;
params.m3   = 0.100  * body_mass;
params.mHAT = 0;
params.I_HAT = 0.678 * body_mass * (0.245 * body_height)^2;
params.L1   = 0.055  * body_height;
params.L2   = 0.246  * body_height;
params.L3   = 0.245  * body_height;
params.d1   = 0.500  * params.L1;
params.d2   = 0.433  * params.L2;
params.d3   = 0.433  * params.L3;
params.I1   = params.m1 * (0.475 * params.L1)^2;
params.I2   = params.m2 * (0.302 * params.L2)^2;
params.I3   = params.m3 * (0.323 * params.L3)^2;

%% ?? Initial Conditions ???????????????????????????????????????????????
X0 = [-0.1; 0.6; -1.1; 0; 0; 0];

%% Zero torque test — model should just slowly fall under gravity
ctrl_zero.tau_max = [0; 0; 0];
ctrl_zero.t_on    = [0.05; 0.03; 0.01];
ctrl_zero.t_off   = [0.35; 0.33; 0.30];
ctrl_zero.k       = 15;

opts_zero = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',1e-4);
[t_z, X_z] = ode45(@(t,X) jump_ode_phase1(t,X,params,ctrl_zero),...
                   [0, 0.5], X0, opts_zero);

fprintf('\n--- Zero torque test ---\n');
fprintf('Simulation ran to t = %.4f s\n', t_z(end));
fprintf('Final phi1=%.1f, phi2=%.1f, phi3=%.1f deg\n',...
    rad2deg(X_z(end,1)),...
    rad2deg(X_z(end,1)+X_z(end,2)),...
    rad2deg(X_z(end,1)+X_z(end,2)+X_z(end,3)));

figure('Name','Zero Torque Test');
subplot(2,1,1);
plot(t_z, rad2deg(X_z(:,1:3)));
legend('\theta_1','\theta_2','\theta_3');
ylabel('Angle (deg)'); title('Joint Angles — Zero Torque'); grid on;
subplot(2,1,2);
plot(t_z, X_z(:,4:6));
legend('\omega_1','\omega_2','\omega_3');
ylabel('Angular vel (rad/s)'); xlabel('Time (s)');
title('Angular Velocities — Zero Torque'); grid on;

%% ?? Check M at Initial Conditions ???????????????????????????????????
phi1=X0(1); phi2=X0(1)+X0(2); phi3=X0(1)+X0(2)+X0(3);
fprintf('Initial absolute angles: phi1=%.2f, phi2=%.2f, phi3=%.2f rad\n',...
    phi1, phi2, phi3);
fprintf('phi1-phi2 = %.2f rad,  phi2-phi3 = %.2f rad\n',...
    phi1-phi2, phi2-phi3);

m1=params.m1; m2=params.m2; m3=params.m3;
L1=params.L1; L2=params.L2; L3=params.L3;
d1=params.d1; d2=params.d2; d3=params.d3;
I1=params.I1; I2=params.I2; I3=params.I3;
I_HAT=params.I_HAT;

c12=cos(phi1-phi2); c13=cos(phi1-phi3); c23=cos(phi2-phi3);
M11=I1+m1*d1^2+(m2+m3)*L1^2;
M22=I2+m2*d2^2+m3*L2^2;
M33=I3+m3*d3^2+I_HAT;
M12=(m2*d2+m3*L2)*L1*c12;
M13=m3*d3*L1*c13;
M23=m3*d3*L2*c23;
M_mat=[M11,M12,M13; M12,M22,M23; M13,M23,M33];
fprintf('Condition number of M: %.2f\n', cond(M_mat));
fprintf('det(M) = %.6f\n\n', det(M_mat));

%% ?? Torque Scale Sweep ???????????????????????????????????????????????
% Target: max vy ? 2-3 m/s at liftoff ? jump height ? 0.2-0.46 m
fprintf('--- Torque scaling sweep ---\n');
fprintf('%-10s %-15s %-15s %-10s\n', 'Scale','Max vy (m/s)','Final phi1 (deg)','Crashed?');

base_tau = [20; 50; 40];   % base torques to scale

for scale = [20, 25, 30, 35, 40, 50]

    ctrl.tau_max = scale * base_tau;
    ctrl.t_on    = [0.05; 0.03; 0.01];
    ctrl.t_off   = [0.35; 0.33; 0.30];
    ctrl.k       = 15;

    opts = odeset('RelTol',1e-3,'AbsTol',1e-4,'MaxStep',1e-3,...
              'NormControl','on','Events',@angle_limit_event);
    try
        [t,X] = ode45(@(t,X) jump_ode_phase1(t,X,params,ctrl),...
                      [0, 1.0], X0, opts);

        % Compute vy at each timestep inline (no external helpers)
        vy = zeros(length(t),1);
        M_tot = m1+m2+m3;
        for i = 1:length(t)
            p1=X(i,1); p2=X(i,1)+X(i,2); p3=X(i,1)+X(i,2)+X(i,3);
            dp1=X(i,4); dp2=X(i,4)+X(i,5); dp3=X(i,4)+X(i,5)+X(i,6);
            vy1=-d1*dp1*sin(p1);
            vy2=-L1*dp1*sin(p1)-d2*dp2*sin(p2);
            vy3=-L1*dp1*sin(p1)-L2*dp2*sin(p2)-d3*dp3*sin(p3);
            vy(i)=(m1*vy1+m2*vy2+m3*vy3)/M_tot;
        end

        crashed = (length(t) < 50) || (max(abs(X(:,1))) > 2*pi);
        fprintf('%-10.4f %-15.4f %-15.1f %-10s\n', scale, max(vy), ...
                rad2deg(X(end,1)), string(crashed));

    catch
        fprintf('%-10.4f %-15s\n', scale, 'ERROR');
    end
end

%% ?? Detailed Run at a Chosen Scale ??????????????????????????????????
% Once you see a scale that gives vy ? 2-3 m/s above, set it here:
chosen_scale = 20;   % <-- adjust this after reading sweep output

ctrl.tau_max = chosen_scale * base_tau;
ctrl.t_on    = [0.05; 0.03; 0.01];
ctrl.t_off   = [0.35; 0.33; 0.30];
ctrl.k       = 15;

opts = odeset('RelTol',1e-3,'AbsTol',1e-4,'MaxStep',1e-3,'NormControl','on','Events',@angle_limit_event);
[t, X] = ode45(@(t,X) jump_ode_phase1(t,X,params,ctrl), [0,1.0], X0, opts);

% Compute vy for detailed plot
vy = zeros(length(t),1);
M_tot = m1+m2+m3;
for i = 1:length(t)
    p1=X(i,1); p2=X(i,1)+X(i,2); p3=X(i,1)+X(i,2)+X(i,3);
    dp1=X(i,4); dp2=X(i,4)+X(i,5); dp3=X(i,4)+X(i,5)+X(i,6);
    vy1=-d1*dp1*sin(p1);
    vy2=-L1*dp1*sin(p1)-d2*dp2*sin(p2);
    vy3=-L1*dp1*sin(p1)-L2*dp2*sin(p2)-d3*dp3*sin(p3);
    vy(i)=(m1*vy1+m2*vy2+m3*vy3)/M_tot;
end

fprintf('\n--- Detailed run at scale = %.4f ---\n', chosen_scale);
fprintf('Simulation ran to t = %.4f s\n', t(end));
fprintf('Max vy = %.4f m/s\n', max(vy));
fprintf('Estimated jump height = %.4f m (%.1f cm)\n', ...
        max(vy)^2/(2*params.g), max(vy)^2/(2*params.g)*100);
fprintf('Final angles (deg): phi1=%.1f, phi2=%.1f, phi3=%.1f\n', ...
        rad2deg(X(end,1)), rad2deg(X(end,1)+X(end,2)), ...
        rad2deg(X(end,1)+X(end,2)+X(end,3)));

figure('Name','Detailed Diagnostic');
subplot(2,1,1);
plot(t, vy, 'g-', 'LineWidth', 1.5);
yline(0,'r--','LineWidth',1.5);
ylabel('vy COM (m/s)'); title('Vertical COM Velocity'); grid on;

subplot(2,1,2);
plot(t, rad2deg(X(:,1:3)));
legend('\theta_1','\theta_2','\theta_3');
ylabel('Angle (deg)'); xlabel('Time (s)');
title('Joint Angles'); grid on;

function [val, isterminal, direction] = angle_limit_event(t, X)  %#ok<INUSL>
    val        = (pi/2 - 0.17) - X(1);
    isterminal = 1;
    direction  = -1;
end