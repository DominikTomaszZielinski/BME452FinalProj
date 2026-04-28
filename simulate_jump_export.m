function [t1, X1, tau_hist, te1, t2, X2, jump_height, liftoff_info] = ...
         simulate_jump_export(params_vec, params)
% Called by validate.py via matlab.engine

if nargin < 2
    % Build default params if not provided
    body_mass=70; body_height=1.75;
    params.g=9.81;
    params.m1=0.0145*body_mass; params.m2=0.0465*body_mass; params.m3=0.100*body_mass;
    params.I_HAT=0.678*body_mass*(0.1)^2;
    params.L1=0.055*body_height; params.L2=0.246*body_height; params.L3=0.245*body_height;
    params.d1=0.500*params.L1; params.d2=0.433*params.L2; params.d3=0.433*params.L3;
    params.I1=params.m1*(0.475*params.L1)^2;
    params.I2=params.m2*(0.302*params.L2)^2;
    params.I3=params.m3*(0.323*params.L3)^2;
end

params_vec = double(params_vec(:));
theta0=params_vec(1:3); tau_max=params_vec(4:6);
t_on=params_vec(7:9);   t_dur=params_vec(10:12);

ctrl.tau_max=tau_max; ctrl.t_on=t_on; ctrl.t_dur=t_dur; ctrl.k=50;
X0_p1=[theta0;0;0;0];

opts1=odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',1e-4,...
    'Events',@(t,X) liftoff_ev(t,X,params,ctrl));
[t1,X1,te1_vec]=ode15s(@(t,X) jump_ode_phase1(t,X,params,ctrl),[0,1.0],X0_p1,opts1);
te1=te1_vec(1);

tau_hist=zeros(length(t1),3);
for i=1:length(t1)
    tau_hist(i,:)=torques(t1(i),ctrl)';
end

X_lo=X1(end,:)';
[x_lo,y_lo]=com_pos(X_lo,params);
[vx_lo,vy_lo]=com_vel(X_lo,params);
liftoff_info=[x_lo,y_lo,vx_lo,vy_lo];

X0_p2=[x_lo;y_lo;X_lo(1:3);vx_lo;vy_lo;0;0;0];
opts2=odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',1e-3,'Events',@land_ev);
[t2,X2]=ode45(@(t,X) jump_ode_phase2(t,X,params),[te1,te1+2.0],X0_p2,opts2);
jump_height=max(X2(:,2))-y_lo;
end

function [v,ist,dir]=liftoff_ev(t,X,params,ctrl)
    dX=jump_ode_phase1(t,X,params,ctrl); ddth=dX(4:6);
    a1=X(1);a2=X(1)+X(2);a3=X(1)+X(2)+X(3);
    dp1=X(4);dp2=X(4)+X(5);dp3=X(4)+X(5)+X(6);
    ddp1=ddth(1);ddp2=ddth(1)+ddth(2);ddp3=ddth(1)+ddth(2)+ddth(3);
    m1=params.m1;m2=params.m2;m3=params.m3;
    L1=params.L1;L2=params.L2;d1=params.d1;d2=params.d2;d3=params.d3;
    Mt=m1+m2+m3;
    ay1=-ddp1*d1*sin(a1)-dp1^2*d1*cos(a1);
    ay2=-ddp1*L1*sin(a1)-dp1^2*L1*cos(a1)-ddp2*d2*sin(a2)-dp2^2*d2*cos(a2);
    ay3=-ddp1*L1*sin(a1)-dp1^2*L1*cos(a1)-ddp2*L2*sin(a2)-dp2^2*L2*cos(a2)...
        -ddp3*d3*sin(a3)-dp3^2*d3*cos(a3);
    v=Mt*(params.g+(m1*ay1+m2*ay2+m3*ay3)/Mt); ist=1; dir=-1;
end

function [v,ist,dir]=land_ev(t,X) %#ok<INUSL>
    v=X(2)-0.1; ist=1; dir=-1;
end

function [x,y]=com_pos(X,p)
    a1=X(1);a2=X(1)+X(2);a3=X(1)+X(2)+X(3);
    M=p.m1+p.m2+p.m3;
    x=(p.m1*p.d1*sin(a1)+p.m2*(p.L1*sin(a1)+p.d2*sin(a2))+p.m3*(p.L1*sin(a1)+p.L2*sin(a2)+p.d3*sin(a3)))/M;
    y=(p.m1*p.d1*cos(a1)+p.m2*(p.L1*cos(a1)+p.d2*cos(a2))+p.m3*(p.L1*cos(a1)+p.L2*cos(a2)+p.d3*cos(a3)))/M;
end

function [vx,vy]=com_vel(X,p)
    a1=X(1);a2=X(1)+X(2);a3=X(1)+X(2)+X(3);
    dp1=X(4);dp2=X(4)+X(5);dp3=X(4)+X(5)+X(6);
    M=p.m1+p.m2+p.m3;
    vx=(p.m1*p.d1*dp1*cos(a1)+p.m2*(p.L1*dp1*cos(a1)+p.d2*dp2*cos(a2))+p.m3*(p.L1*dp1*cos(a1)+p.L2*dp2*cos(a2)+p.d3*dp3*cos(a3)))/M;
    vy=-(p.m1*p.d1*dp1*sin(a1)+p.m2*(p.L1*dp1*sin(a1)+p.d2*dp2*sin(a2))+p.m3*(p.L1*dp1*sin(a1)+p.L2*dp2*sin(a2)+p.d3*dp3*sin(a3)))/M;
end

function tau=torques(t,ctrl)
    k=ctrl.k;
    son=1./(1+exp(-k.*(t-ctrl.t_on)));
    sof=1./(1+exp(-k.*(t-(ctrl.t_on+ctrl.t_dur))));
    tau=[-1;-1;1].*ctrl.tau_max.*(son-sof);
end