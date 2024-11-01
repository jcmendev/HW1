% clc
clear all
close all
rng(123)
%%
% Define agents and goods
n = 3; % Number of agents
m = 3; % Number of goods
% Define parameters
aalpha = 1/m*ones(n,m)            ; % Taste for good j of agent i Scale in utility function
% aalpha = eye(n)+0.25           ; % Taste for good j of agent i Scale in utility function
omega  = -2*ones(n,m)             ; % CRRA Parameter
% omega  = repmat([repmat(-2,n-1,1);-4],1,m)           ; % CRRA Parameter
% Define Endowments
e = rand(n,m);
e = e./sum(e);
% Create a handle with the individual-good utility
u_ij = @(c_ij,alpha_ij,omega_ij) alpha_ij.*(c_ij.^(1+omega_ij))./(1+omega_ij)         ; % 
% Build the planner's problem using symbolic toolbox
ind_good_store = [];
p_good_store   = [];

for in = 1:n
    for jm = 1:m
       ind_good = ['x_',num2str(in),num2str(jm)];
       eval(['syms ',ind_good]) 
       aux_ev = eval(ind_good);
       f_sym(in,jm)      = u_ij(aux_ev,aalpha(in,jm),omega(in,jm))   ;       
       ind_good_store = [ind_good_store,aux_ev] ;         
    end
end
p_good_store   = [];
for jm = 1:m
    p_good = ['p',num2str(jm)];
    eval(['syms ',p_good]) 
    aux_ev_p = eval(p_good);
    p_good_store  = [p_good_store;aux_ev_p] ; 
end
LM_good_store   = [];
for jm = 1:m
    p_good = ['LM',num2str(jm)];
    eval(['syms ',p_good]) 
    aux_ev_p = eval(p_good);
    LM_good_store  = [LM_good_store;aux_ev_p] ; 
end

% CE's problem
pf_sym      = sum(sum(f_sym))         ;
p_good_store = [p_good_store]
%% Build the system
% Resource constraints
MCC = sum([reshape(ind_good_store,m,n),-sum(e)'],2);
% Optimal conditions
grad_pf_sym = [gradient(pf_sym, ind_good_store)]; 
OC = reshape(grad_pf_sym,m,n);
% OC = OC(2:end,:)./OC(1,:);
OC = permute(LM_good_store,[2,1]).*p_good_store-OC;
OC = OC(:);

% Build the BC
BC = [p_good_store].*reshape(ind_good_store,m,n)-[p_good_store].*(e');
BC = sum(BC);
BC = BC(:);
Syst = [OC;BC;MCC];

%% Solve the non-linear system
aux1 = permute(p_good_store(1:end),[2,1]);
aux2 = permute(LM_good_store,[2,1]);
CE_problem = matlabFunction(Syst,'Vars', {[ind_good_store,aux1,aux2]});
x0   = ones(1,m*n+m+n)*1/m;
options = optimoptions('fsolve', 'MaxFunctionEvaluations', 50000, 'Display', 'iter');
[xce_sol,fval,exitflag,output] = fsolve(CE_problem,x0,options);

%% Report
alloc = xce_sol(1:m*n);
ral = reshape(alloc,m,n);
prices = xce_sol(m*n+1:m*n+3);
LMs    = xce_sol(m*n+4:end);

%% Report
columnNames = arrayfun(@(i) sprintf('Agent%d', i), 1:n, 'UniformOutput', false);
rowNames    = arrayfun(@(i) sprintf('Good%d', i), 1:m, 'UniformOutput', false);
T = array2table(ral, 'VariableNames', columnNames, 'RowNames', rowNames);
disp(T)
T2 = array2table([prices]', 'RowNames', rowNames, 'VariableNames', {'Prices'});
disp(T2)

