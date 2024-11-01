% clc
clear all
close all
rng(123)
%% Description:
% This code solves a Social Planner problem of an endowment economy 
% and n and m goods
% 
% JCMV 2024
%%
% Define agents and goods
n = 8; % Number of agents
m = 8; % Number of goods
% Define parameters
% lambda = repmat(1/n*ones(n,1),1,m); % Pareto Weights
lambda = repmat([0.5;0.5/(n-1)*ones(n-1,1)],1,m); % Pareto Weights
% aalpha = 1/m*ones(n,m)            ; % Taste for good j of agent i Scale in utility function
aalpha = eye(n)+0.25           ; % Taste for good j of agent i Scale in utility function
% omega  = -2*ones(n,m)             ; % CRRA Parameter
omega  = repmat([repmat(-2,n-1,1);-4],1,m)           ; % CRRA Parameter
% Define Endowments
e = rand(n,m);
e = e./sum(e);
% Create a handle with the individual-good utility
u_ij = @(c_ij,alpha_ij,omega_ij) alpha_ij.*(c_ij.^(1+omega_ij))./(1+omega_ij)         ; % 
% Build the planner's problem using symbolic toolbox
ind_good_store = [];
for in = 1:n
    for jm = 1:m
       ind_good = ['x_',num2str(in),num2str(jm)];
       eval(['syms ',ind_good]) 
       aux_ev = eval(ind_good);
       f_sym(in,jm)      = u_ij(aux_ev,aalpha(in,jm),omega(in,jm))   ;       
       ind_good_store = [ind_good_store,aux_ev] ;
    end
end
% Planner's problem
pf_sym      = sum(sum(f_sym.*lambda))             ;

%% Solve using FMINCON
% Objective Function
pf   = matlabFunction(-pf_sym,'Vars', {ind_good_store});
% Market clearing Condition
Aeq  = repmat(eye(m),1,n);
beq  = sum(e)';

x0   = ones(m*n,1);
options = optimoptions('fmincon','Display','iter');%,'Algorithm','sqp');
[x,fval] = fmincon(pf,x0',[],[],Aeq,beq,zeros(n*m,1),[],[],options);

%% Report
xr = (reshape(x,m,n));
columnNames = arrayfun(@(i) sprintf('Agent%d', i), 1:n, 'UniformOutput', false);
rowNames    = arrayfun(@(i) sprintf('Good%d', i), 1:m, 'UniformOutput', false);
T = array2table(xr, 'VariableNames', columnNames, 'RowNames', rowNames);
disp(T)

Te = array2table(e', 'VariableNames', columnNames, 'RowNames', rowNames);
disp(Te)


