%% animate_jump.m
%  Animation script for 3-segment planar jump model
%
%  Draws a stick figure (lines + joint dots) across both push-off and
%  flight phases, plays it live in a MATLAB figure, and saves a .gif.
%
%  REQUIREMENTS:
%    - run_simulation.m must be run first (or called from here)
%    - jump_ode_phase1.m and jump_ode_phase2.m must be on the path
%
%  OUTPUT:
%    - Live animation in figure window
%    - jump_animation.gif saved in the current directory
clear; clc; close all;

%% 1. RUN SIMULATION TO GET TRAJECTORY DATA 
%  We re-run the simulation here so animate_jump.m is self-contained.
%  If you already have t1,X1,t2,X2,params in your workspace, comment
%  out this section and jump to Section 2.

fprintf('Running simulation...\n');

% parameters (must match run_simulation.m exactly)
body_mass   = 70;
body_height = 1.75;

params.g    = 9.81;
params.m1   = 0.0145 * body_mass;
params.m2   = 0.0465 * body_mass;
params.m3   = 0.100  * body_mass;
% HAT treated as added inertia at hip, not a point mass above it
% This avoids the massive destabilizing moment arm
params.mHAT  = 0;                        % remove point mass
params.I_HAT = 0.678 * body_mass * (0.1)^2; % add as rotational inertia at hip using thigh length as radius
params.L1   = 0.055  * body_height;
params.L2   = 0.246  * body_height;
params.L3   = 0.245  * body_height;
params.d1   = 0.500  * params.L1;
params.d2   = 0.433  * params.L2;
params.d3   = 0.433  * params.L3;
params.I1   = params.m1 * (0.475 * params.L1)^2;
params.I2   = params.m2 * (0.302 * params.L2)^2;
params.I3   = params.m3 * (0.323 * params.L3)^2;

% Control parameters -- step-function torques, fixed at literature maxima
% ankle 175 N.m (Hussain & Frey-Law 2016), knee 247 N.m, hip 175 N.m (Harbo 2012)
ctrl.tau_max = [175; 247; 175];   % N.m

% Initial conditions
% phi1=0.3 ? foot tilts forward slightly
% phi2=0.3+(-0.3)=0.0 ? shank near vertical
% phi3=0.0+0.8=0.8 ? thigh bent forward ~46 deg
% At these 3 conditions, liftoff not detected. 0.0, 1.2, and -1.8 b4
% Optimal starting angles from Bayesian Optimization (clean round, 78.62 cm)
% theta1 (ankle absolute) : -0.6000 rad
% theta2 (knee relative)  :  1.8000 rad
% theta3 (hip relative)   : -1.3836 rad
X0_p1 = [-0.6000; 1.8000; -1.3836; 0; 0; 0];

t_span1 = [0, 1.0];   % allow up to 0.6 s for push-off

% Phase 1: toe-pinned
opts1 = odeset(...
    'RelTol',    1e-6,  ...
    'AbsTol',    1e-8, ...
    'Events',    @(t,X) liftoff_event(t, X, params, ctrl), ...
    'MaxStep',   1e-4);
[t1, X1, te1, ~, ~] = ode15s(@(t,X) jump_ode_phase1(t, X, params, ctrl), ...
    t_span1, X0_p1, opts1);

if isempty(te1)
    error('Liftoff not detected. Check torques or initial conditions.');
end
fprintf('Liftoff at t = %.4f s\n', te1);

% Liftoff & Phase 2 handoff 
X_lo = X1(end,:)';
[x_com_lo, y_com_lo] = compute_com_position(X_lo, params);
[vx_com_lo, vy_com_lo] = compute_com_velocity(X_lo, params);

% Keep angle trajectories
%X0_p2 = [x_com_lo; y_com_lo; X_lo(1:3); vx_com_lo; vy_com_lo; X_lo(4:6)];

% Zero out angular velocities in flight for cleaner animation
% Segments hold liftoff pose during flight (reasonable simplification)
X0_p2 = [x_com_lo; y_com_lo; X_lo(1:3); vx_com_lo; vy_com_lo; 0; 0; 0];

% Phase 2: free flight
opts2 = odeset('RelTol',1e-8,'AbsTol',1e-10,...
    'Events', @(t,X) peak_height_event(t,X),'MaxStep',1e-3);
[t2, X2] = ode45(@(t,X) jump_ode_phase2(t,X,params),...
    [te1, te1+2.0], X0_p2, opts2);

jump_height = max(X2(:,2)) - y_com_lo;
fprintf('Jump height = %.4f m (%.2f cm)\n', jump_height, jump_height*100);


%% 1b. PLOT INITIAL STANDING CONFIGURATION
fig_init = figure('Name','Initial Squat Configuration','Color','w',...
                  'Position',[950 100 400 600]);
ax_init = axes('Parent', fig_init);
hold(ax_init, 'on');
grid(ax_init, 'on');
axis(ax_init, 'equal');
xlabel(ax_init, 'x (m)');
ylabel(ax_init, 'y (m)');
title(ax_init, 'Initial Squat Posture');

% Compute joint positions at t=0
jp0 = get_joint_positions_p1(X0_p1, params);
x_toe0=jp0(1,1); y_toe0=jp0(1,2);
x_ank0=jp0(2,1); y_ank0=jp0(2,2);
x_kne0=jp0(3,1); y_kne0=jp0(3,2);
x_hip0=jp0(4,1); y_hip0=jp0(4,2);

% Draw segments
col_s = [0.15 0.45 0.75];
col_j = [0.90 0.30 0.20];
line(ax_init,[x_toe0,x_ank0],[y_toe0,y_ank0],'Color',col_s,'LineWidth',3);
line(ax_init,[x_ank0,x_kne0],[y_ank0,y_kne0],'Color',col_s,'LineWidth',3);
line(ax_init,[x_kne0,x_hip0],[y_kne0,y_hip0],'Color',col_s,'LineWidth',3);

% Draw joints
plot(ax_init,x_toe0,y_toe0,'o','Color',col_j,'MarkerFaceColor',col_j,'MarkerSize',10);
plot(ax_init,x_ank0,y_ank0,'o','Color',col_j,'MarkerFaceColor',col_j,'MarkerSize',10);
plot(ax_init,x_kne0,y_kne0,'o','Color',col_j,'MarkerFaceColor',col_j,'MarkerSize',10);
plot(ax_init,x_hip0,y_hip0,'o','Color',col_j,'MarkerFaceColor',col_j,'MarkerSize',10);

% Labels
text(ax_init, x_toe0+0.02, y_toe0,      'Toe',   'FontSize',10);
text(ax_init, x_ank0+0.02, y_ank0,      'Ankle', 'FontSize',10);
text(ax_init, x_kne0+0.02, y_kne0,      'Knee',  'FontSize',10);
text(ax_init, x_hip0+0.02, y_hip0,      'Hip',   'FontSize',10);

% Ground line
yline(ax_init, 0, 'k-', 'LineWidth', 2);

% Annotate angles
[~, y_com0] = compute_com_position(X0_p1, params);
text(ax_init, x_toe0-0.20, y_com0, ...
    sprintf('\\theta_1=%.1f° (toe)\n\\theta_2=%.1f°(ankle)\n\\theta_3=%.1f° (knee)', ...
    rad2deg(X0_p1(1)), rad2deg(X0_p1(2)), rad2deg(X0_p1(3))), ...
    'FontSize', 9, 'Color', [0.3 0.3 0.3]);

% Draw COM
[x_com0, y_com0] = compute_com_position(X0_p1, params);
plot(ax_init, x_com0, y_com0, 'p', ...
    'Color', [0.9 0.6 0.0], ...
    'MarkerFaceColor', [0.9 0.6 0.0], ...
    'MarkerSize', 14, ...
    'DisplayName', 'COM');
text(ax_init, x_com0+0.02, y_com0+0.04, 'COM', 'FontSize', 10, 'Color', [0.9 0.6 0.0]);

% Draw vertical dashed line from COM to ground (to show height)
line(ax_init, [x_com0, x_com0], [0, y_com0], ...
    'Color', [0.9 0.6 0.0], 'LineStyle', '--', 'LineWidth', 1.0);
text(ax_init, x_com0+0.02, y_com0/2, ...
    sprintf('h = %.3f m', y_com0), ...
    'FontSize', 9, 'Color', [0.9 0.6 0.0]);

% Padding
xlim(ax_init, [-0.3, 0.5]);
ylim(ax_init, [-0.05, max([y_hip0, y_kne0, y_ank0])+0.15]);

%% 2. BUILD UNIFIED TIMELINE 
%  Resample both phases onto a common fixed-timestep grid for smooth
%  animation. 120 fps gives smooth playback; reduce if slow on your machine.

fps        = 60;
dt_anim    = 1/fps;
t_end      = t2(end);
t_anim     = 0 : dt_anim : t_end;
n_frames   = length(t_anim);

% Interpolate phase 1 (only valid up to liftoff)
t1_mask = t_anim <= te1;
t2_mask = t_anim >  te1;

% Phase-1 frames: interpolate X1 onto animation timeline
X1_anim = interp1(t1, X1, t_anim(t1_mask), 'pchip');

% Phase-2 frames: interpolate X2 onto animation timeline
X2_anim = interp1(t2, X2, t_anim(t2_mask), 'pchip');

fprintf('Total frames: %d  (%.1f s at %d fps)\n', n_frames, t_end, fps);


%% 3. PRE-COMPUTE JOINT POSITIONS FOR EVERY FRAME 
%  This avoids recomputing geometry inside the animation loop.

joint_pos = zeros(n_frames, 5, 2);
% Columns: [toe, ankle, knee, hip, HAT_point] x [x, y]

% Phase 1 frames (toe pinned at origin)
for i = 1:sum(t1_mask)
    Xi = X1_anim(i,:)';
    joint_pos(i,:,:) = get_joint_positions_p1(Xi, params);
end

% Phase 2 frames (COM translates freely)
idx_p1 = sum(t1_mask);
for i = 1:sum(t2_mask)
    Xi = X2_anim(i,:)';
    joint_pos(idx_p1+i,:,:) = get_joint_positions_p2(Xi, params);
end

% Pre-compute COM position for every frame (for trace line)
com_pos = zeros(n_frames, 2);
for i = 1:sum(t1_mask)
    Xi = X1_anim(i,:)';
    [cx, cy] = compute_com_position(Xi, params);
    com_pos(i,:) = [cx, cy];
end
for i = 1:sum(t2_mask)
    Xi = X2_anim(i,:)';
    com_pos(idx_p1+i,:) = [Xi(1), Xi(2)];
end

%% 4. SET UP FIGURE 

fig = figure('Name','Jump Animation','Color','w',...
             'Position',[100 100 800 700]);

% Determine axis limits from full trajectory
all_x = joint_pos(:,:,1);
all_y = joint_pos(:,:,2);
x_pad = 0.3;
y_pad = 0.1;
ax_xlim = [min(all_x(:))-x_pad,  max(all_x(:))+x_pad];
%ax_ylim = [-30, 1];
ax_ylim = [-0.05,                 max(all_y(:))+y_pad];

ax = axes('Parent', fig);
axis(ax, [ax_xlim, ax_ylim]);
axis(ax, 'equal');
hold(ax, 'on');
grid(ax, 'on');
xlabel(ax, 'x (m)');
ylabel(ax, 'y (m)');

% Ground line
yline(ax, 0, 'k-', 'LineWidth', 2);

% Colour scheme
col_seg   = [0.15 0.45 0.75];   % segment lines  (blue)
col_joint = [0.90 0.30 0.20];   % joint dots     (red)
col_HAT   = [0.20 0.65 0.30];   % HAT point      (green)
col_phase = [0.85 0.85 0.85];   % phase label bg

% Pre-create graphics objects (update their data each frame)
h_foot  = line(ax, [0 0],[0 0],'Color',col_seg,'LineWidth',3);
h_shank = line(ax, [0 0],[0 0],'Color',col_seg,'LineWidth',3);
h_thigh = line(ax, [0 0],[0 0],'Color',col_seg,'LineWidth',3);

h_toe   = plot(ax, 0,0,'o','Color',col_joint,'MarkerFaceColor',col_joint,'MarkerSize',8);
h_ankle = plot(ax, 0,0,'o','Color',col_joint,'MarkerFaceColor',col_joint,'MarkerSize',8);
h_knee  = plot(ax, 0,0,'o','Color',col_joint,'MarkerFaceColor',col_joint,'MarkerSize',8);
h_hip   = plot(ax, 0,0,'o','Color',col_joint,'MarkerFaceColor',col_joint,'MarkerSize',8);
h_HAT   = plot(ax, 0,0,'p','Color',[0.9 0.6 0.0],'MarkerFaceColor',[0.9 0.6 0.0],'MarkerSize',14,'DisplayName','COM');

% COM trajectory trace
h_trace = plot(ax, NaN, NaN, '--', 'Color', [0.6 0.2 0.8], 'LineWidth', 1.5);
com_trace_x = zeros(1, n_frames);
com_trace_y = zeros(1, n_frames);

h_time  = title(ax, 't = 0.000 s   |   Phase: Push-off');
h_label = text(ax, ax_xlim(1)+0.05, ax_ylim(2)-0.05, '',...
               'FontSize',11,'FontWeight','bold','Color',[0.5 0.1 0.1]);


%% 5. GIF SETUP 

gif_filename = 'jump_animation.gif';
gif_delay    = dt_anim;   % seconds per frame in gif


%% 6. ANIMATION LOOP 

fprintf('Animating... (close figure to stop early)\n');

for i = 1:n_frames

    if ~ishandle(fig), break; end   % user closed window

    % Joint positions this frame
    jp = squeeze(joint_pos(i,:,:));   % [5 x 2]: toe,ankle,knee,hip,HAT

    x_toe   = jp(1,1);  y_toe   = jp(1,2);
    x_ankle = jp(2,1);  y_ankle = jp(2,2);
    x_knee  = jp(3,1);  y_knee  = jp(3,2);
    x_hip   = jp(4,1);  y_hip   = jp(4,2);
    x_HAT   = jp(5,1);  y_HAT   = jp(5,2);

    % Update segment lines
    set(h_foot,  'XData',[x_toe,   x_ankle], 'YData',[y_toe,   y_ankle]);
    set(h_shank, 'XData',[x_ankle, x_knee],  'YData',[y_ankle, y_knee]);
    set(h_thigh, 'XData',[x_knee,  x_hip],   'YData',[y_knee,  y_hip]);

    % Update joint dots
    set(h_toe,   'XData', x_toe,   'YData', y_toe);
    set(h_ankle, 'XData', x_ankle, 'YData', y_ankle);
    set(h_knee,  'XData', x_knee,  'YData', y_knee);
    set(h_hip,   'XData', x_hip,   'YData', y_hip);
    set(h_HAT, 'XData', com_pos(i,1), 'YData', com_pos(i,2));

    % Update COM trace (show all points up to current frame)
    com_trace_x(i) = com_pos(i,1);
    com_trace_y(i) = com_pos(i,2);
    set(h_trace, 'XData', com_trace_x(1:i), 'YData', com_trace_y(1:i));
    
    % Phase label and title
    if t_anim(i) <= te1
        phase_str = 'Push-off';
    else
        phase_str = 'Flight';
    end
    set(h_time, 'String', sprintf('t = %.3f s   |   Phase: %s', t_anim(i), phase_str));
    set(h_label,'String', phase_str);

    drawnow limitrate;

    % Capture frame for GIF
    frame     = getframe(fig);
    [img, cm] = rgb2ind(frame2im(frame), 256);

    if i == 1
        imwrite(img, cm, gif_filename, 'gif', ...
            'LoopCount', Inf, 'DelayTime', gif_delay);
    else
        imwrite(img, cm, gif_filename, 'gif', ...
            'WriteMode','append','DelayTime', gif_delay);
    end

end

fprintf('Animation complete.\n');
fprintf('GIF saved to: %s\n', fullfile(pwd, gif_filename));

%% 7. PHASE 1 ONLY ANIMATION (push-off debugging)

fig2 = figure('Name','Push-off Phase Only','Color','w',...
              'Position',[100 100 800 700]);
ax2 = axes('Parent', fig2);

% Use same axis limits but focused on ground level
xlim(ax2, [-0.5, 0.5]);
ylim(ax2, [-0.05, 1.2]);
axis(ax2, 'equal');
hold(ax2, 'on');
grid(ax2, 'on');
xlabel(ax2, 'x (m)');
ylabel(ax2, 'y (m)');
yline(ax2, 0, 'k-', 'LineWidth', 2);

% Graphics objects
h2_foot  = line(ax2,[0 0],[0 0],'Color',col_seg,'LineWidth',3);
h2_shank = line(ax2,[0 0],[0 0],'Color',col_seg,'LineWidth',3);
h2_thigh = line(ax2,[0 0],[0 0],'Color',col_seg,'LineWidth',3);
h2_toe   = plot(ax2,0,0,'o','Color',col_joint,'MarkerFaceColor',col_joint,'MarkerSize',8);
h2_ankle = plot(ax2,0,0,'o','Color',col_joint,'MarkerFaceColor',col_joint,'MarkerSize',8);
h2_knee  = plot(ax2,0,0,'o','Color',col_joint,'MarkerFaceColor',col_joint,'MarkerSize',8);
h2_hip   = plot(ax2,0,0,'o','Color',col_joint,'MarkerFaceColor',col_joint,'MarkerSize',8);
h2_com   = plot(ax2,0,0,'p','Color',[0.9 0.6 0.0], 'MarkerFaceColor',[0.9 0.6 0.0],'MarkerSize',14,'DisplayName','COM');
h2_trace = plot(ax2,NaN,NaN,'--','Color',[0.6 0.2 0.8],'LineWidth',1.5);
h2_title = title(ax2,'t = 0.000 s  |  Push-off');

% COM trace storage
n_p1 = sum(t1_mask);
p1_trace_x = zeros(1, n_p1);
p1_trace_y = zeros(1, n_p1);

gif2_filename = 'pushoff_animation.gif';

fprintf('Animating push-off phase only...\n');

for i = 1:n_p1
    if ~ishandle(fig2), break; end

    jp = squeeze(joint_pos(i,:,:));
    x_toe=jp(1,1); y_toe=jp(1,2);
    x_ank=jp(2,1); y_ank=jp(2,2);
    x_kne=jp(3,1); y_kne=jp(3,2);
    x_hip=jp(4,1); y_hip=jp(4,2);

    set(h2_foot, 'XData',[x_toe,x_ank],'YData',[y_toe,y_ank]);
    set(h2_shank,'XData',[x_ank,x_kne],'YData',[y_ank,y_kne]);
    set(h2_thigh,'XData',[x_kne,x_hip],'YData',[y_kne,y_hip]);
    set(h2_toe,  'XData',x_toe,'YData',y_toe);
    set(h2_ankle,'XData',x_ank,'YData',y_ank);
    set(h2_knee, 'XData',x_kne,'YData',y_kne);
    set(h2_hip,  'XData',x_hip,'YData',y_hip);

    % COM
    cx = com_pos(i,1); cy = com_pos(i,2);
    set(h2_com,'XData',cx,'YData',cy);
    p1_trace_x(i) = cx; p1_trace_y(i) = cy;
    set(h2_trace,'XData',p1_trace_x(1:i),'YData',p1_trace_y(1:i));

    set(h2_title,'String',sprintf('t = %.3f s  |  Push-off  (liftoff at t=%.3f s)',...
        t_anim(i), te1));

    drawnow limitrate;

    % Save to gif
    frame = getframe(fig2);
    [img,cm] = rgb2ind(frame2im(frame),256);
    if i==1
        imwrite(img,cm,gif2_filename,'gif','LoopCount',Inf,'DelayTime',dt_anim);
    else
        imwrite(img,cm,gif2_filename,'gif','WriteMode','append','DelayTime',dt_anim);
    end
end

fprintf('Push-off animation saved to: %s\n', gif2_filename);
%%  GEOMETRY HELPER FUNCTIONS
%  Returns [toe; ankle; knee; hip; HAT] positions as [5x2] matrix.

function jp = get_joint_positions_p1(X, params)
% Phase 1: toe fixed at origin, joints built upward from there.

    phi1 = X(1);
    phi2 = X(1) + X(2);
    phi3 = X(1) + X(2) + X(3);

    L1=params.L1; L2=params.L2; L3=params.L3;

    x_toe   = 0;  y_toe   = 0;
    x_ankle = x_toe   + L1*sin(phi1);   y_ankle = y_toe   + L1*cos(phi1);
    x_knee  = x_ankle + L2*sin(phi2);   y_knee  = y_ankle + L2*cos(phi2);
    x_hip   = x_knee  + L3*sin(phi3);   y_hip   = y_knee  + L3*cos(phi3);
    x_HAT   = x_hip;                    y_HAT   = y_hip;   % HAT = point at hip

    jp = [x_toe,   y_toe;
          x_ankle, y_ankle;
          x_knee,  y_knee;
          x_hip,   y_hip;
          x_HAT,   y_HAT];
end


function jp = get_joint_positions_p2(X, params)
% Phase 2: COM translates; reconstruct joints relative to COM offset.
%
% Strategy:
%   1. Compute where the COM sits relative to toe (using phase-1 geometry
%      at the liftoff angles, stored in X(3:5)).
%   2. Add the actual COM displacement X(1:2) to get absolute positions.

    x_com = X(1);  y_com = X(2);
    phi1  = X(3);
    phi2  = X(3) + X(4);
    phi3  = X(3) + X(4) + X(5);

    L1=params.L1; L2=params.L2; L3=params.L3;
    d1=params.d1; d2=params.d2; d3=params.d3;
    m1=params.m1; m2=params.m2; m3=params.m3;
    M_tot = m1+m2+m3;

    % Joint positions relative to toe (toe at arbitrary reference 0,0)
    ax_ = L1*sin(phi1);              ay_ = L1*cos(phi1);
    kx_ = ax_ + L2*sin(phi2);       ky_ = ay_ + L2*cos(phi2);
    hx_ = kx_ + L3*sin(phi3);       hy_ = ky_ + L3*cos(phi3);

    % COM relative to toe — no HAT point mass, matches jump_ode_phase1.m
    x1_=d1*sin(phi1);                          y1_=d1*cos(phi1);
    x2_=L1*sin(phi1)+d2*sin(phi2);             y2_=L1*cos(phi1)+d2*cos(phi2);
    x3_=L1*sin(phi1)+L2*sin(phi2)+d3*sin(phi3); y3_=L1*cos(phi1)+L2*cos(phi2)+d3*cos(phi3);

    xc_ref=(m1*x1_+m2*x2_+m3*x3_)/M_tot;
    yc_ref=(m1*y1_+m2*y2_+m3*y3_)/M_tot;

    % Offset: shift so COM matches X(1:2)
    dx = x_com - xc_ref;
    dy = y_com - yc_ref;

    x_toe   = 0   + dx;  y_toe   = 0   + dy;   % toe floats in flight
    x_ankle = ax_ + dx;  y_ankle = ay_ + dy;
    x_knee  = kx_ + dx;  y_knee  = ky_ + dy;
    x_hip   = hx_ + dx;  y_hip   = hy_ + dy;
    x_HAT   = x_hip;     y_HAT   = y_hip;

    jp = [x_toe,   y_toe;
          x_ankle, y_ankle;
          x_knee,  y_knee;
          x_hip,   y_hip;
          x_HAT,   y_HAT];
end


%%  EVENT FUNCTIONS  (duplicated from run_simulation.m for self-containment)

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


%%  COM HELPERS  (duplicated for self-containment)

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
    dp1=X(4); dp2=X(4)+X(5); dp3=X(4)+X(5)+X(6);
    ddp1=ddtheta(1); ddp2=ddtheta(1)+ddtheta(2); ddp3=ddtheta(1)+ddtheta(2)+ddtheta(3);
    m1=params.m1; m2=params.m2; m3=params.m3; mHAT=0;
    L1=params.L1; L2=params.L2; L3=params.L3;
    d1=params.d1; d2=params.d2; d3=params.d3;
    M_tot=m1+m2+m3+mHAT;
    ay1=-ddp1*d1*sin(phi1)-dp1^2*d1*cos(phi1);
    ay2=-ddp1*L1*sin(phi1)-dp1^2*L1*cos(phi1)-ddp2*d2*sin(phi2)-dp2^2*d2*cos(phi2);
    ay3=-ddp1*L1*sin(phi1)-dp1^2*L1*cos(phi1)-ddp2*L2*sin(phi2)-dp2^2*L2*cos(phi2)...
        -ddp3*d3*sin(phi3)-dp3^2*d3*cos(phi3);
    ayH=-ddp1*L1*sin(phi1)-dp1^2*L1*cos(phi1)-ddp2*L2*sin(phi2)-dp2^2*L2*cos(phi2)...
        -ddp3*L3*sin(phi3)-dp3^2*L3*cos(phi3);
    ay_com=(m1*ay1+m2*ay2+m3*ay3+mHAT*ayH)/M_tot;
end

function tau = compute_torques(t, ctrl)  %#ok<INUSL>
% STEP FUNCTION: full torque applied immediately from t=0 to liftoff.
    tau_sign = [-1; -1; 1];
    tau      = tau_sign .* ctrl.tau_max;
end
