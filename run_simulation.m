%% run_simulation.m
%  Main simulation script for 3-segment planar jump model
%
%  Runs two sequential ODE phases:
%    Phase 1 - Toe-pinned push-off  (ends at liftoff)
%    Phase 2 - Free-body flight     (ends at peak COM height)
%
%  Returns jump_height (m) above the COM height at liftoff.
%  This scalar is the objective BOTorch will maximise later.
%
%  References:
%    Winter (2009) anthropometric parameters
%    Bobbert & van Soest (1994) jump model structure
clear; clc; close all;

%% 1. MODEL PARAMETERS  (Winter 2009, 70 kg / 1.75 m male)

body_mass   = 70;    % kg
body_height = 1.75;  % m

params.g = 9.81;     % m/s^2

% Segment masses (fraction of total body mass)
params.m1 = 0.0145 * body_mass;   % foot   (1.015 kg)
params.m2 = 0.0465 * body_mass;   % shank  (3.255 kg)
params.m3 = 0.100  * body_mass;   % thigh  (7.000 kg)
params.mHAT  = 0;
params.I_HAT = 0.678 * body_mass * (0.1)^2;

% Segment lengths (fraction of body height)
params.L1 = 0.055  * body_height;  % foot   (0.096 m)
params.L2 = 0.246  * body_height;  % shank  (0.431 m)
params.L3 = 0.245  * body_height;  % thigh  (0.429 m)

% COM location from proximal end (fraction of segment length)
params.d1 = 0.500  * params.L1;   % foot
params.d2 = 0.433  * params.L2;   % shank
params.d3 = 0.433  * params.L3;   % thigh

% Moment of inertia about COM: I = m * (r_gyr * L)^2
% Radius of gyration fractions from Winter (2009)
params.I1 = params.m1 * (0.475 * params.L1)^2;  % foot
params.I2 = params.m2 * (0.302 * params.L2)^2;  % shank
params.I3 = params.m3 * (0.323 * params.L3)^2;  % thigh

fprintf('=== Model Parameters ===\n');
fprintf('Foot:  m=%.3f kg, L=%.3f m, I=%.5f kg.m^2\n', params.m1, params.L1, params.I1);
fprintf('Shank: m=%.3f kg, L=%.3f m, I=%.5f kg.m^2\n', params.m2, params.L2, params.I2);
fprintf('Thigh: m=%.3f kg, L=%.3f m, I=%.5f kg.m^2\n', params.m3, params.L3, params.I3);
fprintf('HAT:   m=%.3f kg (point mass at hip)\n', params.mHAT);
fprintf('\n');


%% 2. CONTROL PARAMETERS  (starting guess — BOTorch tunes these)
%
%  Sigmoidal torque profile per joint:
%    tau_i(t) = tau_max_i * [sig(t, t_on_i) - sig(t, t_on_i + t_dur_i)]
%
%  12 optimisable parameters:
%    theta0(3)  : initial joint angles
%    tau_max(3) : peak torques
%    t_on(3)    : activation onset times
%    t_dur(3)   : activation durations
%  k is a fixed sharpness constant (not optimised).
%
%  Joint order: [ankle (1), knee (2), hip (3)]

ctrl.tau_max = [600; 1500; 1200];
ctrl.t_on    = [0.05; 0.04; 0.03];
ctrl.t_dur   = [0.30; 0.30; 0.30];   % duration instead of t_off
ctrl.k       = 50;

% Quick sanity check: t_on must be before t_off for each joint
assert(all(ctrl.t_dur > 0), 'Control error: t_dur must be positive.');


%% 3. INITIAL CONDITIONS  (crouched squat posture)
%
%  theta0 = [theta1, theta2, theta3] — also optimised by BOTorch
%  Bounds: theta1 in [-1.5, 0], theta2 in [0.5, 2.5], theta3 in [-2.5, -0.5]
%  All angular velocities start at zero.

theta0 = [-0.1; 0.6; -1.1];   % current baseline — BOTorch will tune this
X0_p1  = [theta0; 0; 0; 0];

%% ENERGY CHECK AT t=0
phi1=X0_p1(1); phi2=X0_p1(1)+X0_p1(2); phi3=X0_p1(1)+X0_p1(2)+X0_p1(3);

% Potential energy
PE = params.g*(params.m1*params.d1*cos(phi1) ...
             + params.m2*(params.L1*cos(phi1)+params.d2*cos(phi2)) ...
             + params.m3*(params.L1*cos(phi1)+params.L2*cos(phi2)+params.d3*cos(phi3)));
fprintf('Initial PE = %.4f J\n', PE);

% Initial gravity generalized forces (G vector)
g=params.g; m1=params.m1; m2=params.m2; m3=params.m3;
L1=params.L1; L2=params.L2; L3=params.L3;
d1=params.d1; d2=params.d2; d3=params.d3;

G1 = g*(m1*d1*sin(phi1)+m2*(L1*sin(phi1)+d2*sin(phi2))+m3*(L1*sin(phi1)+L2*sin(phi2)+d3*sin(phi3)));
G2 = g*(m2*d2*sin(phi2)+m3*(L2*sin(phi2)+d3*sin(phi3)));
G3 = g*m3*d3*sin(phi3);
fprintf('G vector at t=0: [%.4f, %.4f, %.4f] N.m\n', G1, G2, G3);

% Initial torques at t=0
tau0 = [-1;-1;1] .* ctrl.tau_max .* (1./(1+exp(-ctrl.k*(0-ctrl.t_on))) - 1./(1+exp(-ctrl.k*(0-(ctrl.t_on+ctrl.t_dur)))));
fprintf('Tau at t=0: [%.6f, %.6f, %.6f] N.m\n', tau0(1), tau0(2), tau0(3));

% What ddtheta does M\(tau-G) give at t=0?
I_HAT=params.I_HAT; I1=params.I1; I2=params.I2; I3=params.I3;
M11=(I1+m1*d1^2)+(I2+m2*d2^2)+(I3+m3*d3^2+I_HAT)+m2*L1^2+m3*L1^2+m3*L2^2 ...
    +2*m2*L1*d2*cos(phi2-phi1)+2*m3*L1*L2*cos(phi2-phi1) ...
    +2*m3*L1*d3*cos(phi3-phi1)+2*m3*L2*d3*cos(phi3-phi2);
M22=(I2+m2*d2^2)+(I3+m3*d3^2+I_HAT)+m3*L2^2+2*m3*L2*d3*cos(phi3-phi2);
M33=I3+m3*d3^2+I_HAT;
M12=(I2+m2*d2^2)+(I3+m3*d3^2+I_HAT)+m3*L2^2 ...
    +m2*L1*d2*cos(phi2-phi1)+m3*L1*L2*cos(phi2-phi1) ...
    +m3*L1*d3*cos(phi3-phi1)+2*m3*L2*d3*cos(phi3-phi2);
M13=(I3+m3*d3^2+I_HAT)+m3*L1*d3*cos(phi3-phi1)+m3*L2*d3*cos(phi3-phi2);
M23=(I3+m3*d3^2+I_HAT)+m3*L2*d3*cos(phi3-phi2);
M_mat=[M11 M12 M13; M12 M22 M23; M13 M23 M33];

G_vec = [G1;G2;G3];
ddtheta0_grav = M_mat \ (-G_vec);
ddtheta0_tau  = M_mat \ (tau0 - G_vec);
fprintf('ddtheta from gravity only: [%.2f, %.2f, %.2f] rad/s^2\n', ddtheta0_grav);
fprintf('ddtheta from tau+gravity:  [%.2f, %.2f, %.2f] rad/s^2\n', ddtheta0_tau);
fprintf('cond(M) = %.2f\n', cond(M_mat));

% Check M condition number at liftoff angle
phi1_lo = pi/2 - 0.17;
phi2_lo = phi1_lo + X0_p1(2);
phi3_lo = phi2_lo + X0_p1(3);
M11b=(I1+m1*d1^2)+(I2+m2*d2^2)+(I3+m3*d3^2+I_HAT)+m2*L1^2+m3*L1^2+m3*L2^2 ...
    +2*m2*L1*d2*cos(phi2_lo-phi1_lo)+2*m3*L1*L2*cos(phi2_lo-phi1_lo) ...
    +2*m3*L1*d3*cos(phi3_lo-phi1_lo)+2*m3*L2*d3*cos(phi3_lo-phi2_lo);
M22b=(I2+m2*d2^2)+(I3+m3*d3^2+I_HAT)+m3*L2^2+2*m3*L2*d3*cos(phi3_lo-phi2_lo);
M33b=I3+m3*d3^2+I_HAT;
M12b=(I2+m2*d2^2)+(I3+m3*d3^2+I_HAT)+m3*L2^2 ...
    +m2*L1*d2*cos(phi2_lo-phi1_lo)+m3*L1*L2*cos(phi2_lo-phi1_lo) ...
    +m3*L1*d3*cos(phi3_lo-phi1_lo)+2*m3*L2*d3*cos(phi3_lo-phi2_lo);
M13b=(I3+m3*d3^2+I_HAT)+m3*L1*d3*cos(phi3_lo-phi1_lo)+m3*L2*d3*cos(phi3_lo-phi2_lo);
M23b=(I3+m3*d3^2+I_HAT)+m3*L2*d3*cos(phi3_lo-phi2_lo);
M_lo=[M11b M12b M13b; M12b M22b M23b; M13b M23b M33b];
fprintf('cond(M) at liftoff angles: %.2f\n', cond(M_lo));
fprintf('det(M) at liftoff angles:  %.6f\n', det(M_lo));
fprintf('M at liftoff:\n'); disp(M_lo);

% Verify initial COM is above ground (y > 0)
[~, y_com_init] = compute_com_position(X0_p1, params);
assert(y_com_init > 0, 'Initial COM is below ground. Check initial angles.');
fprintf('Initial COM height: %.4f m\n', y_com_init);

% Check NEW Jacobian mass matrix condition number
q_test = X0_p1(1:3);
a1=q_test(1); a2=q_test(1)+q_test(2); a3=q_test(1)+q_test(2)+q_test(3);
J1n=[d1*cos(a1),0,0; -d1*sin(a1),0,0];
J2n=[L1*cos(a1)+d2*cos(a2),d2*cos(a2),0; -L1*sin(a1)-d2*sin(a2),-d2*sin(a2),0];
J3n=[L1*cos(a1)+L2*cos(a2)+d3*cos(a3),L2*cos(a2)+d3*cos(a3),d3*cos(a3);
    -L1*sin(a1)-L2*sin(a2)-d3*sin(a3),-L2*sin(a2)-d3*sin(a3),-d3*sin(a3)];
Jw1n=[1,0,0]; Jw2n=[1,1,0]; Jw3n=[1,1,1];
M_new = params.m1*(J1n'*J1n)+params.I1*(Jw1n'*Jw1n)+ ...
        params.m2*(J2n'*J2n)+params.I2*(Jw2n'*Jw2n)+ ...
        params.m3*(J3n'*J3n)+(params.I3+params.I_HAT)*(Jw3n'*Jw3n);
fprintf('NEW Jacobian M condition number: %.2f\n', cond(M_new));
fprintf('NEW M:\n'); disp(M_new);
ddq_grav_new = M_new \ (-[G1;G2;G3]);
fprintf('NEW ddtheta from gravity only: [%.2f, %.2f, %.2f] rad/s^2\n', ddq_grav_new);

% Compute COM Jacobian y-row to find correct torque signs
a1=phi1; a2=phi2; a3=phi3;
Jcom_y = -(m1*d1*sin(a1) + m2*(L1*sin(a1)+d2*sin(a2)) + ...
            m3*(L1*sin(a1)+L2*sin(a2)+d3*sin(a3))) / (m1+m2+m3);
Jcom_y2= -(m2*d2*sin(a2) + m3*(L2*sin(a2)+d3*sin(a3))) / (m1+m2+m3);
Jcom_y3= -(m3*d3*sin(a3)) / (m1+m2+m3);
fprintf('COM Jacobian y-row: [%.4f, %.4f, %.4f]\n', Jcom_y, Jcom_y2, Jcom_y3);
fprintf('For upward COM motion, torques should have same sign as Jcom_y entries\n');

%% 4. PHASE 1: TOE-PINNED PUSH-OFF

t_span1 = [0, 1.0];   % allow up to 0.6 s for push-off

opts1 = odeset(...
    'RelTol',    1e-6,  ...
    'AbsTol',    1e-8, ...
    'Events',    @(t,X) liftoff_event(t, X, params, ctrl), ...
    'MaxStep',   1e-4);

[t1, X1, te1, ~, ~] = ode15s(...
    @(t,X) jump_ode_phase1(t, X, params, ctrl), ...
    t_span1, X0_p1, opts1);

if isempty(te1)
    warning('Liftoff event not detected. Model may not be jumping.');
    warning('Check torque magnitudes or initial posture.');
    jump_height = 0;
    return;
end

fprintf('Liftoff detected at t = %.4f s\n', te1);
fprintf('Theta1 at liftoff: %.4f rad (%.1f deg)\n', X1(end,1), rad2deg(X1(end,1)));
fprintf('Theta1 range during push-off: %.4f to %.4f rad\n', min(X1(:,1)), max(X1(:,1)));
fprintf('Max angular velocity dtheta1: %.4f rad/s\n', max(abs(X1(:,4))));
fprintf('Max angular velocity dtheta2: %.4f rad/s\n', max(abs(X1(:,5))));
fprintf('Max angular velocity dtheta3: %.4f rad/s\n', max(abs(X1(:,6))));

% Check vy at liftoff
[~, vy_check] = compute_com_velocity(X1(end,:)', params);
fprintf('vy at liftoff: %.4f m/s\n', vy_check);
fprintf('theta1 at liftoff: %.4f rad (%.1f deg)\n', X1(end,1), rad2deg(X1(end,1)));

% Plot COM velocity during push-off
figure('Name','COM velocity during push-off');
vy_hist = zeros(length(t1),1);
for ii=1:length(t1)
    [~,vy_hist(ii)] = compute_com_velocity(X1(ii,:)', params);
end
plot(t1, vy_hist, 'b-', 'LineWidth', 2);
yline(0,'r--');
xlabel('Time (s)'); ylabel('vy COM (m/s)');
title('Vertical COM velocity during push-off');
grid on;

%% 5. LIFTOFF STATE & PHASE 2 INITIAL CONDITIONS 

X_lo = X1(end, :)';   % state at liftoff

% COM position and velocity at liftoff (from phase-1 geometry)
[x_com_lo, y_com_lo] = compute_com_position(X_lo, params);
[vx_com_lo, vy_com_lo] = compute_com_velocity(X_lo, params);

% Sanity check: COM should be moving upward at liftoff
if vy_com_lo <= 0
    warning('COM vertical velocity at liftoff is <= 0 (%.4f m/s).', vy_com_lo);
    warning('Model is not jumping upward. Check torques or initial conditions.');
end

fprintf('Liftoff COM position: (%.4f, %.4f) m\n', x_com_lo, y_com_lo);
fprintf('Liftoff COM velocity: (%.4f, %.4f) m/s\n', vx_com_lo, vy_com_lo);

% Phase-2 state: [x_com; y_com; theta1; theta2; theta3;
%                 vx_com; vy_com; dtheta1; dtheta2; dtheta3]
X0_p2 = [x_com_lo; y_com_lo; X_lo(1:3); vx_com_lo; vy_com_lo; X_lo(4:6)];
% segment angles carry over unchanged & angular velocities carry over unchanged

% Zero out angular velocities in flight for cleaner animation
% Segments hold liftoff pose during flight (reasonable simplification)
% X0_p2 = [x_com_lo; y_com_lo; X_lo(1:3); vx_com_lo; vy_com_lo; 0; 0; 0];

%% 6. PHASE 2: FREE-BODY FLIGHT 

t_span2 = [te1, te1 + 2.0];   % allow up to 2 s of flight

opts2 = odeset(...
    'RelTol',    1e-8,  ...
    'AbsTol',    1e-10, ...
    'Events',    @(t,X) peak_height_event(t, X), ...
    'MaxStep',   1e-3);

[t2, X2, te2, ~, ~] = ode45(...
    @(t,X) jump_ode_phase2(t, X, params), ...
    t_span2, X0_p2, opts2);

if isempty(te2)
    warning('Peak height event not detected during flight phase.');
    % Fall back: use maximum y_com in trajectory
    jump_height = max(X2(:,2)) - y_com_lo;
else
    jump_height = max(X2(:,2)) - y_com_lo;
end


%% 7. RESULTS 

fprintf('\n=== Simulation Results ===\n');
fprintf('Jump height:     %.4f m  (%.2f cm)\n', jump_height, jump_height*100);
fprintf('Peak COM height: %.4f m\n', max(X2(:,2)));
fprintf('Flight time:     %.4f s\n', t2(end) - te1);
fprintf('\n');


%% 8. BASIC DIAGNOSTIC PLOTS 

figure('Name','Phase 1 — Joint Angles During Push-off');
subplot(3,1,1); plot(t1, rad2deg(X1(:,1))); ylabel('\theta_1 (deg)'); title('Foot angle'); grid on;
subplot(3,1,2); plot(t1, rad2deg(X1(:,2))); ylabel('\theta_2 (deg)'); title('Knee angle (relative)'); grid on;
subplot(3,1,3); plot(t1, rad2deg(X1(:,3))); ylabel('\theta_3 (deg)'); title('Hip angle (relative)'); grid on;
xlabel('Time (s)');

figure('Name','Phase 1 — Torque Profiles');
t_plot = linspace(0, te1, 200)';
tau_plot = zeros(length(t_plot), 3);
for i = 1:length(t_plot)
    tau_plot(i,:) = compute_torques(t_plot(i), ctrl)';
end
plot(t_plot, tau_plot); legend('Ankle','Knee','Hip'); 
xlabel('Time (s)'); ylabel('Torque (Nm)'); title('Joint Torques During Push-off'); grid on;

figure('Name','Phase 2 — COM Trajectory During Flight');
plot(X2(:,1), X2(:,2), 'b-', 'LineWidth', 2);
hold on;
plot(x_com_lo, y_com_lo, 'go', 'MarkerSize', 10, 'DisplayName', 'Liftoff');
plot(X2(end,1), X2(end,2), 'r*', 'MarkerSize', 10, 'DisplayName', 'Peak');
xlabel('x (m)'); ylabel('y (m)'); title('COM Trajectory (Flight Phase)');
legend('Trajectory','Liftoff','Peak'); axis equal; grid on;


%%  EVENT FUNCTIONS
%  (defined here as local functions so run_simulation.m is self-contained)

function [val, isterminal, direction] = liftoff_event(t, X, params, ctrl)
% Liftoff when vertical COM velocity becomes positive
% i.e. the model is actually moving upward
    dX      = jump_ode_phase1(t, X, params, ctrl);
    ddtheta = dX(4:6);
    a1=X(1); a2=X(1)+X(2); a3=X(1)+X(2)+X(3);
    dp1=X(4); dp2=X(4)+X(5); dp3=X(4)+X(5)+X(6);
    ddp1=ddtheta(1); ddp2=ddtheta(1)+ddtheta(2); ddp3=ddtheta(1)+ddtheta(2)+ddtheta(3);
    m1=params.m1; m2=params.m2; m3=params.m3;
    L1=params.L1; L2=params.L2;
    d1=params.d1; d2=params.d2; d3=params.d3;
    M_tot=m1+m2+m3;
    ay1=-ddp1*d1*sin(a1)-dp1^2*d1*cos(a1);
    ay2=-ddp1*L1*sin(a1)-dp1^2*L1*cos(a1)-ddp2*d2*sin(a2)-dp2^2*d2*cos(a2);
    ay3=-ddp1*L1*sin(a1)-dp1^2*L1*cos(a1)-ddp2*L2*sin(a2)-dp2^2*L2*cos(a2) ...
        -ddp3*d3*sin(a3)-dp3^2*d3*cos(a3);
    ay_com=(m1*ay1+m2*ay2+m3*ay3)/M_tot;
    GRF_y = M_tot*(params.g + ay_com);
    val        = GRF_y;
    isterminal = 1;
    direction  = -1;
end


function [val, isterminal, direction] = peak_height_event(t, X)  %#ok<INUSL>
% Stop when model lands (y_com returns to liftoff height) or 2s elapsed
    val=X(2) - 0.1;   % stop when COM drops below 0.1m (near ground)
    isterminal=1;
    direction=-1;
end


%%  WRAPPER CALLS TO HELPER FUNCTIONS IN jump_ode_phase1.m
%  MATLAB requires these to be accessible; they are defined there.
%  The wrappers below simply forward the calls.

function [x_com, y_com] = compute_com_position(X, params)
    a1=X(1); a2=X(1)+X(2); a3=X(1)+X(2)+X(3);
    m1=params.m1; m2=params.m2; m3=params.m3;
    L1=params.L1; L2=params.L2;
    d1=params.d1; d2=params.d2; d3=params.d3;
    M_tot=m1+m2+m3;
    x1=d1*sin(a1);                                y1=d1*cos(a1);
    x2=L1*sin(a1)+d2*sin(a2);                     y2=L1*cos(a1)+d2*cos(a2);
    x3=L1*sin(a1)+L2*sin(a2)+d3*sin(a3);          y3=L1*cos(a1)+L2*cos(a2)+d3*cos(a3);
    x_com=(m1*x1+m2*x2+m3*x3)/M_tot;
    y_com=(m1*y1+m2*y2+m3*y3)/M_tot;
end

function [vx_com, vy_com] = compute_com_velocity(X, params)
    a1=X(1); a2=X(1)+X(2); a3=X(1)+X(2)+X(3);
    dp1=X(4); dp2=X(4)+X(5); dp3=X(4)+X(5)+X(6);
    m1=params.m1; m2=params.m2; m3=params.m3;
    L1=params.L1; L2=params.L2;
    d1=params.d1; d2=params.d2; d3=params.d3;
    M_tot=m1+m2+m3;
    vx1= d1*dp1*cos(a1);                          vy1=-d1*dp1*sin(a1);
    vx2= L1*dp1*cos(a1)+d2*dp2*cos(a2);           vy2=-L1*dp1*sin(a1)-d2*dp2*sin(a2);
    vx3= L1*dp1*cos(a1)+L2*dp2*cos(a2)+d3*dp3*cos(a3);
    vy3=-L1*dp1*sin(a1)-L2*dp2*sin(a2)-d3*dp3*sin(a3);
    vx_com=(m1*vx1+m2*vx2+m3*vx3)/M_tot;
    vy_com=(m1*vy1+m2*vy2+m3*vy3)/M_tot;
end

function ay_com = compute_com_accel_y(X, ddtheta, params)
    phi1=X(1); phi2=X(1)+X(2); phi3=X(1)+X(2)+X(3);
    dp1=X(4);  dp2=X(4)+X(5);  dp3=X(4)+X(5)+X(6);
    ddp1=ddtheta(1); ddp2=ddtheta(1)+ddtheta(2); ddp3=ddtheta(1)+ddtheta(2)+ddtheta(3);
    m1=params.m1; m2=params.m2; m3=params.m3; mHAT=0;
    L1=params.L1; L2=params.L2; L3=params.L3;
    d1=params.d1; d2=params.d2; d3=params.d3;
    M_tot=m1+m2+m3+mHAT;

    ay1= -ddp1*d1*sin(phi1) - dp1^2*d1*cos(phi1);
    ay2= -ddp1*L1*sin(phi1) - dp1^2*L1*cos(phi1) - ddp2*d2*sin(phi2) - dp2^2*d2*cos(phi2);
    ay3= -ddp1*L1*sin(phi1) - dp1^2*L1*cos(phi1) - ddp2*L2*sin(phi2) - dp2^2*L2*cos(phi2) ...
         -ddp3*d3*sin(phi3) - dp3^2*d3*cos(phi3);
    ayH= -ddp1*L1*sin(phi1) - dp1^2*L1*cos(phi1) - ddp2*L2*sin(phi2) - dp2^2*L2*cos(phi2) ...
         -ddp3*L3*sin(phi3) - dp3^2*L3*cos(phi3);

    ay_com=(m1*ay1+m2*ay2+m3*ay3+mHAT*ayH)/M_tot;
end

function tau = compute_torques(t, ctrl)
    k        = ctrl.k;
    sig_on   = 1./(1+exp(-k.*(t - ctrl.t_on)));
    sig_off  = 1./(1+exp(-k.*(t - (ctrl.t_on + ctrl.t_dur))));
    tau_sign = [-1; -1; 1];
    tau      = tau_sign .* ctrl.tau_max .* (sig_on - sig_off);
end

function jump_height = simulate_jump(params_vec, params)
% SIMULATE_JUMP  Wrapper for BOTorch optimisation
%
% params_vec: 12-element vector
%   [1:3]  theta0   - initial joint angles (rad)
%   [4:6]  tau_max  - peak torques (Nm)
%   [7:9]  t_on     - activation onset times (s)
%   [10:12] t_dur   - activation durations (s)
%
% Returns jump_height (m) — the scalar BOTorch maximises.

% Unpack
theta0  = params_vec(1:3);
tau_max = params_vec(4:6);
t_on    = params_vec(7:9);
t_dur   = params_vec(10:12);

% Build ctrl
ctrl.tau_max = tau_max;
ctrl.t_on    = t_on;
ctrl.t_dur   = t_dur;
ctrl.k       = 50;

% Build initial conditions
X0_p1 = [theta0; 0; 0; 0];

% Feasibility check — COM must be above ground
[~, y_com_init] = compute_com_position(X0_p1, params);
if y_com_init <= 0
    jump_height = 0;
    return;
end

% Phase 1: push-off
opts1 = odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1e-4,...
    'Events',@(t,X) liftoff_event(t,X,params,ctrl));
try
    [t1,X1,te1] = ode15s(@(t,X) jump_ode_phase1(t,X,params,ctrl),...
        [0,1.0], X0_p1, opts1);
catch
    jump_height = 0; return;
end

if isempty(te1) || te1 < 0.005
    jump_height = 0; return;
end

% Liftoff state
X_lo = X1(end,:)';
[x_com_lo, y_com_lo] = compute_com_position(X_lo, params);
[vx_com_lo, vy_com_lo] = compute_com_velocity(X_lo, params);

if vy_com_lo <= 0
    jump_height = 0; return;
end

% Phase 2: flight
X0_p2 = [x_com_lo; y_com_lo; X_lo(1:3); vx_com_lo; vy_com_lo; 0; 0; 0];
opts2 = odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',1e-3);
try
    [~,X2] = ode45(@(t,X) jump_ode_phase2(t,X,params),...
        [te1, te1+2.0], X0_p2, opts2);
catch
    jump_height = 0; return;
end

jump_height = max(X2(:,2)) - y_com_lo;
end

function [t1, X1, tau_hist, te1, t2, X2, jump_height, liftoff_info] = ...
         simulate_jump_export(params_vec, params)
% Called by validate.py via matlab.engine

params_vec = double(params_vec(:));

theta0  = params_vec(1:3);
tau_max = params_vec(4:6);
t_on    = params_vec(7:9);
t_dur   = params_vec(10:12);

ctrl.tau_max = tau_max;
ctrl.t_on    = t_on;
ctrl.t_dur   = t_dur;
ctrl.k       = 50;

X0_p1 = [theta0; 0; 0; 0];

% Phase 1
opts1 = odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1e-4,...
    'Events',@(t,X) liftoff_event_exp(t,X,params,ctrl));
[t1,X1,te1_vec] = ode15s(@(t,X) jump_ode_phase1(t,X,params,ctrl),...
    [0,1.0], X0_p1, opts1);
te1 = te1_vec(1);

% Torque history
tau_hist = zeros(length(t1), 3);
for i = 1:length(t1)
    tau_hist(i,:) = compute_torques_exp(t1(i), ctrl)';
end

% Liftoff
X_lo = X1(end,:)';
[x_lo, y_lo]   = compute_com_pos_exp(X_lo, params);
[vx_lo, vy_lo] = compute_com_vel_exp(X_lo, params);
liftoff_info   = [x_lo, y_lo, vx_lo, vy_lo];

% Phase 2
X0_p2 = [x_lo; y_lo; X_lo(1:3); vx_lo; vy_lo; 0; 0; 0];
opts2 = odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',1e-3,...
    'Events',@landing_event_exp);
[t2,X2] = ode45(@(t,X) jump_ode_phase2(t,X,params),...
    [te1, te1+2.0], X0_p2, opts2);

jump_height = max(X2(:,2)) - y_lo;
end

function [val,ist,dir] = liftoff_event_exp(t,X,params,ctrl)
    dX=jump_ode_phase1(t,X,params,ctrl); ddtheta=dX(4:6);
    a1=X(1);a2=X(1)+X(2);a3=X(1)+X(2)+X(3);
    dp1=X(4);dp2=X(4)+X(5);dp3=X(4)+X(5)+X(6);
    ddp1=ddtheta(1);ddp2=ddtheta(1)+ddtheta(2);ddp3=ddtheta(1)+ddtheta(2)+ddtheta(3);
    m1=params.m1;m2=params.m2;m3=params.m3;
    L1=params.L1;L2=params.L2;d1=params.d1;d2=params.d2;d3=params.d3;
    M_tot=m1+m2+m3;
    ay1=-ddp1*d1*sin(a1)-dp1^2*d1*cos(a1);
    ay2=-ddp1*L1*sin(a1)-dp1^2*L1*cos(a1)-ddp2*d2*sin(a2)-dp2^2*d2*cos(a2);
    ay3=-ddp1*L1*sin(a1)-dp1^2*L1*cos(a1)-ddp2*L2*sin(a2)-dp2^2*L2*cos(a2)...
        -ddp3*d3*sin(a3)-dp3^2*d3*cos(a3);
    ay_com=(m1*ay1+m2*ay2+m3*ay3)/M_tot;
    GRF_y=M_tot*(params.g+ay_com);
    val=GRF_y; ist=1; dir=-1;
end

function [val,ist,dir] = landing_event_exp(t,X)  %#ok<INUSL>
    val=X(2)-0.1; ist=1; dir=-1;
end

function [x,y] = compute_com_pos_exp(X,params)
    a1=X(1);a2=X(1)+X(2);a3=X(1)+X(2)+X(3);
    m1=params.m1;m2=params.m2;m3=params.m3;
    L1=params.L1;L2=params.L2;d1=params.d1;d2=params.d2;d3=params.d3;
    M=m1+m2+m3;
    x=(m1*d1*sin(a1)+m2*(L1*sin(a1)+d2*sin(a2))+m3*(L1*sin(a1)+L2*sin(a2)+d3*sin(a3)))/M;
    y=(m1*d1*cos(a1)+m2*(L1*cos(a1)+d2*cos(a2))+m3*(L1*cos(a1)+L2*cos(a2)+d3*cos(a3)))/M;
end

function [vx,vy] = compute_com_vel_exp(X,params)
    a1=X(1);a2=X(1)+X(2);a3=X(1)+X(2)+X(3);
    dp1=X(4);dp2=X(4)+X(5);dp3=X(4)+X(5)+X(6);
    m1=params.m1;m2=params.m2;m3=params.m3;
    L1=params.L1;L2=params.L2;d1=params.d1;d2=params.d2;d3=params.d3;
    M=m1+m2+m3;
    vx=(m1*d1*dp1*cos(a1)+m2*(L1*dp1*cos(a1)+d2*dp2*cos(a2))+m3*(L1*dp1*cos(a1)+L2*dp2*cos(a2)+d3*dp3*cos(a3)))/M;
    vy=-(m1*d1*dp1*sin(a1)+m2*(L1*dp1*sin(a1)+d2*dp2*sin(a2))+m3*(L1*dp1*sin(a1)+L2*dp2*sin(a2)+d3*dp3*sin(a3)))/M;
end

function tau = compute_torques_exp(t,ctrl)
    k=ctrl.k;
    sig_on=1./(1+exp(-k.*(t-ctrl.t_on)));
    sig_off=1./(1+exp(-k.*(t-(ctrl.t_on+ctrl.t_dur))));
    tau_sign=[-1;-1;1];
    tau=tau_sign.*ctrl.tau_max.*(sig_on-sig_off);
end