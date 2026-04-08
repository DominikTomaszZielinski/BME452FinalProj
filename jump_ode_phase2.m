function dX = jump_ode_phase2(t, X, params)  %#ok<INUSL>
% JUMP_ODE_PHASE2  Free-body flight dynamics (post-liftoff phase)
%
% Once the toe leaves the ground there are no constraint forces and no
% joint torques (muscles have already fired during push-off).  The system
% COM follows a purely ballistic trajectory while the segments carry their
% angular momenta from liftoff.
%
% State vector X = [x_com; y_com; theta1; theta2; theta3;
%                   vx_com; vy_com; dtheta1; dtheta2; dtheta3]
%
%   x_com, y_com   : COM position (m)
%   theta1,2,3     : segment angles, same convention as phase 1
%   vx_com, vy_com : COM velocity (m/s)
%   dtheta1,2,3    : relative angular velocities (rad/s)
%
% Usage:
%   dX = jump_ode_phase2(t, X, params)

g = params.g;

% Unpack state 
% Positions
x_com  = X(1);  %#ok<NASGU>
y_com  = X(2);  %#ok<NASGU>
% theta1 = X(3);  % not needed for flight EOM
% theta2 = X(4);
% theta3 = X(5);

% Velocities
vx_com  = X(6);
vy_com  = X(7);
dtheta  = X(8:10);

% Equations of motion 
% COM: ballistic (gravity only, no ground reaction)
dx_com  = vx_com;
dy_com  = vy_com;
dvx_com = 0;
dvy_com = -g;

% Segments: no joint torques, no ground constraint -> free rotation
% Angular momentum of each segment is conserved individually
% (rigid body, no air resistance)
ddtheta = [0; 0; 0];

% Assemble derivative vector
dX = [dx_com;          % d/dt x_com
      dy_com;          % d/dt y_com
      dtheta;          % d/dt theta1,2,3  (= dtheta, already unpacked)
      dvx_com;         % d/dt vx_com
      dvy_com;         % d/dt vy_com
      ddtheta];        % d/dt dtheta1,2,3
end
