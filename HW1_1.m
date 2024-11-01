clc
clear all
close all
%% Description:
% This code integrates the function 
%      int_{0}^{100} {exp(-rho*t).*(-exp(-(1-exp(-lambda*t))))}dt
% Using quadrature methods: Midpoint, Trapezoid and Simpson and 
% and simulation Monte Carlo methods
% 
% JCMV 2024
%% II. Integration
% Parameters
rho    = 0.04  ;
lambda = 0.02  ;
a      = 0.00  ;
b      = 100.0 ;
n_int  = [10 100 1000 10000 100000];
% Create the functions handle 
c = @(t) (1-exp(-lambda*t))  ; % c(t) 
u = @(c) (-exp(-c))         ; % u(c(t))
f = @(u,t) exp(-rho*t).*u   ; % f(u(c(t))
f2 = @(t) exp(-rho*t).*(-exp(-(1-exp(-lambda*t)))); % f(u(c(t)) directly
nn = length(n_int);
% Integrate with 4 methods
integ_res = zeros(nn,4);
for ixn = 1:nn
    n = n_int(ixn);
    % Midpoint rule
    tic
    mid_h        = (b-a)/n                       ; % Step size
    mid_n_list   = (1:n)'                        ; % List from 1 to n intervals
    mid_ev_grid  = a+(mid_n_list-(1/2))*mid_h    ; % Evaluation of mid integral
    mid_integral = mid_h*sum(f2(mid_ev_grid))   ;
    toc
    tic
    % Trapezoid rule
    trap_h        = (b-a)/n                      ;
    trap_n_list   = (1:n-1)'                     ; % List from 1 to n intervals
    trap_ev_grid  = a+(trap_n_list*trap_h)       ; % 
    trap_integral = trap_h*((f2(a)/2)+sum(f2(trap_ev_grid))+(f2(b)/2));
    mtrap_integral= trapz(linspace(a,b,n),f2(linspace(a,b,n)));
    toc
    tic
    % Simpson
    simp_h        = (b-a)/n                      ;
    simp_n_list   = (1:n-1)'                     ; % List from 1 to n intervals
    simp_ev_grid  = a+(simp_n_list*simp_h)       ; %
    simp_factor   = 4*mod(trap_n_list,2)+2*~mod(trap_n_list,2);
    simp_integral = simp_h/3*(f2(a)+sum(simp_factor.*f2(simp_ev_grid))+f2(b));
    toc
    tic
    % Monte Carlo
    MC_draws      = a + (b-a).*rand(n,1);
    MC_integral   = ((b-a)/n)*sum(f2(MC_draws));
    toc
    % Store results    
    integ_res(ixn,:) = [mid_integral,mtrap_integral,simp_integral,MC_integral];
end

%% Analytic
syms t
f2      = @(t) exp(-rho*t).*(-exp(-(1-exp(-lambda*t)))); % f(u(c(t)) directly
udef_int = int(f2, t);
def_int = int(f2, t, 0, 100);
xval    = double(def_int);
%% Report
format long
xr = integ_res./integ_res(end,1);
xr = ((integ_res./xval)-1)*100;
% xr = integ_res;
columnNames = {'Midpoint','Trapezoid','Simpson','Monte Carlo'};
rowNames    = arrayfun(@(i) sprintf('%d Nodes', n_int(i)), 1:nn, 'UniformOutput', false);
T = array2table(xr, 'VariableNames', columnNames, 'RowNames', rowNames);
disp(T)

%%
