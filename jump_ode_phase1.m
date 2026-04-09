function dX = jump_ode_phase1(t, X, params, ctrl)
% JUMP_ODE_PHASE1 - Toe-pinned 3-segment jump dynamics
% Uses Newton-Euler approach via Jacobians — numerically robust.
%
% Generalized coords q = [theta1, theta2, theta3]
%   theta1: foot absolute angle from vertical
%   theta2: shank relative to foot
%   theta3: thigh relative to shank
% Toe pinned at origin. y positive upward.

q  = X(1:3);
dq = X(4:6);

m1=params.m1; m2=params.m2; m3=params.m3;
L1=params.L1; L2=params.L2; L3=params.L3;
d1=params.d1; d2=params.d2; d3=params.d3;
I1=params.I1; I2=params.I2; I3=params.I3;
I_HAT=params.I_HAT;
g=params.g;

% Absolute angles
a1=q(1); a2=q(1)+q(2); a3=q(1)+q(2)+q(3);

% COM positions (toe at origin)
% Segment 1 (foot): COM at d1 from toe along foot
p1 = [d1*sin(a1);  d1*cos(a1)];
% Segment 2 (shank): ankle + d2 along shank
p2 = [L1*sin(a1)+d2*sin(a2);  L1*cos(a1)+d2*cos(a2)];
% Segment 3 (thigh): knee + d3 along thigh
p3 = [L1*sin(a1)+L2*sin(a2)+d3*sin(a3); ...
      L1*cos(a1)+L2*cos(a2)+d3*cos(a3)];

% Jacobians: dp_i/dq (2x3 matrices)
% Each column k is d(p_i)/d(q_k)
% Since a1=q1, a2=q1+q2, a3=q1+q2+q3:
%   d(sin(a_j))/d(q_k) = cos(a_j) if k<=j, 0 otherwise
%   d(cos(a_j))/d(q_k) = -sin(a_j) if k<=j, 0 otherwise

J1 = [d1*cos(a1),  0,  0;
     -d1*sin(a1),  0,  0];

J2 = [L1*cos(a1)+d2*cos(a2),  d2*cos(a2),  0;
     -L1*sin(a1)-d2*sin(a2), -d2*sin(a2),  0];

J3 = [L1*cos(a1)+L2*cos(a2)+d3*cos(a3), ...
      L2*cos(a2)+d3*cos(a3), ...
      d3*cos(a3); ...
     -L1*sin(a1)-L2*sin(a2)-d3*sin(a3), ...
     -L2*sin(a2)-d3*sin(a3), ...
     -d3*sin(a3)];

% Angular velocity Jacobians (scalar, each segment rotates at abs ang vel)
% dtheta_abs_i/dq_k = 1 if k<=i, 0 otherwise
Jw1 = [1, 0, 0];   % foot absolute angular velocity = dq1
Jw2 = [1, 1, 0];   % shank absolute angular velocity = dq1+dq2
Jw3 = [1, 1, 1];   % thigh absolute angular velocity = dq1+dq2+dq3

% Mass matrix: M = sum_i (m_i*Ji'*Ji + I_i*Jwi'*Jwi)
M = m1*(J1'*J1) + I1*(Jw1'*Jw1) + ...
    m2*(J2'*J2) + I2*(Jw2'*Jw2) + ...
    m3*(J3'*J3) + I3*(Jw3'*Jw3) + ...
    I_HAT*(Jw3'*Jw3);  % HAT adds rotational inertia at hip (same Jw as thigh)

% Coriolis/centripetal: h = dM/dt * dq - 0.5*grad_q(dq'*M*dq)
% Computed via numerical differentiation of M (robust, no hand derivation)
% ?? Coriolis/centripetal via dJ/dt ????????????????????????????????????
% h = sum_i [ m_i*Ji'*(dJi/dt*dq) + I_i*Jwi'*(dJwi/dt*dq) ]
% dJi/dt = sum_k (dJi/dqk)*dqk  — computed analytically below

% Absolute angular velocities
da1=dq(1); da2=dq(1)+dq(2); da3=dq(1)+dq(2)+dq(3);

% Time derivatives of Jacobians (dJ/dt = sum_k dJ/dqk * dqk)
% J1 depends only on a1=q1
dJ1dt = [-d1*sin(a1)*da1,  0,  0;
         -d1*cos(a1)*da1,  0,  0];

% J2 depends on a1=q1 and a2=q1+q2
dJ2dt = [-L1*sin(a1)*da1-d2*sin(a2)*da2,  -d2*sin(a2)*da2,  0;
         -L1*cos(a1)*da1-d2*cos(a2)*da2,  -d2*cos(a2)*da2,  0];

% J3 depends on a1,a2,a3
dJ3dt = [-L1*sin(a1)*da1-L2*sin(a2)*da2-d3*sin(a3)*da3, ...
         -L2*sin(a2)*da2-d3*sin(a3)*da3, ...
         -d3*sin(a3)*da3; ...
         -L1*cos(a1)*da1-L2*cos(a2)*da2-d3*cos(a3)*da3, ...
         -L2*cos(a2)*da2-d3*cos(a3)*da3, ...
         -d3*cos(a3)*da3];

% Angular Jacobians are constant (Jwi = const) so dJwi/dt = 0

% Coriolis vector: h_k = sum_i m_i*(J_i*dq).(dJ_i/dt*dq) ... 
% Full form: h = sum_i [m_i*Ji'*(dJidt*dq) + Ii*Jwi'*(dJwidt*dq)]
h = m1*(J1'*(dJ1dt*dq)) + ...
    m2*(J2'*(dJ2dt*dq)) + ...
    m3*(J3'*(dJ3dt*dq));
% Note: dJwi/dt = 0 so angular terms vanish
% Subtract half the velocity-dependent term (standard Lagrangian form):
% This is already the correct Coriolis vector from the Jacobian formulation
% Gravity: G_k = dV/dq_k, V = sum_i m_i*g*y_i
% y_i = p_i(2), d(cos(a_j))/d(q_k) = -sin(a_j) for k<=j
G = zeros(3,1);
% Segment 1
G(1) = G(1) + m1*g*(-d1*sin(a1));
% Segment 2
G(1) = G(1) + m2*g*(-L1*sin(a1)-d2*sin(a2));
G(2) = G(2) + m2*g*(-d2*sin(a2));
% Segment 3
G(1) = G(1) + m3*g*(-L1*sin(a1)-L2*sin(a2)-d3*sin(a3));
G(2) = G(2) + m3*g*(-L2*sin(a2)-d3*sin(a3));
G(3) = G(3) + m3*g*(-d3*sin(a3));

% Torques
tau = compute_torques(t, ctrl);

% EOM: M*ddq + h + G = tau  ?  ddq = M\(tau - h - G)
ddq = M \ (tau - h - G);

dX = [dq; ddq];
end


function M = mass_matrix(q, m1,m2,m3,L1,L2,L3,d1,d2,d3,I1,I2,I3,I_HAT) %#ok<INUSL>
a1=q(1); a2=q(1)+q(2); a3=q(1)+q(2)+q(3);

J1 = [d1*cos(a1),  0,  0;
     -d1*sin(a1),  0,  0];
J2 = [L1*cos(a1)+d2*cos(a2),  d2*cos(a2),  0;
     -L1*sin(a1)-d2*sin(a2), -d2*sin(a2),  0];
J3 = [L1*cos(a1)+L2*cos(a2)+d3*cos(a3), L2*cos(a2)+d3*cos(a3), d3*cos(a3);
     -L1*sin(a1)-L2*sin(a2)-d3*sin(a3),-L2*sin(a2)-d3*sin(a3),-d3*sin(a3)];

Jw1=[1,0,0]; Jw2=[1,1,0]; Jw3=[1,1,1];

M = m1*(J1'*J1) + I1*(Jw1'*Jw1) + ...
    m2*(J2'*J2) + I2*(Jw2'*Jw2) + ...
    m3*(J3'*J3) + (I3+I_HAT)*(Jw3'*Jw3);
end


function tau = compute_torques(t, ctrl)
k      = ctrl.k;
sig_on = 1./(1+exp(-k.*(t-ctrl.t_on)));
sig_off= 1./(1+exp(-k.*(t-ctrl.t_off)));
tau_sign = [-1; -1; 1];
tau    = tau_sign .* ctrl.tau_max .* (sig_on - sig_off);
end


function [x_com, y_com] = compute_com_position(X, params)
a1=X(1); a2=X(1)+X(2); a3=X(1)+X(2)+X(3);
m1=params.m1; m2=params.m2; m3=params.m3;
L1=params.L1; L2=params.L2;
d1=params.d1; d2=params.d2; d3=params.d3;
M_tot=m1+m2+m3;
y1=d1*cos(a1);
y2=L1*cos(a1)+d2*cos(a2);
y3=L1*cos(a1)+L2*cos(a2)+d3*cos(a3);
x1=d1*sin(a1);
x2=L1*sin(a1)+d2*sin(a2);
x3=L1*sin(a1)+L2*sin(a2)+d3*sin(a3);
x_com=(m1*x1+m2*x2+m3*x3)/M_tot;
y_com=(m1*y1+m2*y2+m3*y3)/M_tot;
end


function [vx_com,vy_com] = compute_com_velocity(X, params)
a1=X(1); a2=X(1)+X(2); a3=X(1)+X(2)+X(3);
dp1=X(4); dp2=X(4)+X(5); dp3=X(4)+X(5)+X(6);
m1=params.m1; m2=params.m2; m3=params.m3;
L1=params.L1; L2=params.L2;
d1=params.d1; d2=params.d2; d3=params.d3;
M_tot=m1+m2+m3;
vx1= d1*dp1*cos(a1);
vy1=-d1*dp1*sin(a1);
vx2= L1*dp1*cos(a1)+d2*dp2*cos(a2);
vy2=-L1*dp1*sin(a1)-d2*dp2*sin(a2);
vx3= L1*dp1*cos(a1)+L2*dp2*cos(a2)+d3*dp3*cos(a3);
vy3=-L1*dp1*sin(a1)-L2*dp2*sin(a2)-d3*dp3*sin(a3);
vx_com=(m1*vx1+m2*vx2+m3*vx3)/M_tot;
vy_com=(m1*vy1+m2*vy2+m3*vy3)/M_tot;
end


function ay_com = compute_com_accel_y(X, ddtheta, params)
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
end