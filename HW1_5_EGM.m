clc
clear all
close all
%% Description:
% This code solves by EGM a Recursive Competitive Equilibrium of
% production economy with tax distortions, adjustment costs of investment
% (In Progress)
% 
% JCMV 2024

%% Technical Parameters 
maxiter = 25000;
tol     = 1e-6 ;
theta_upd   = 0.15 ; % Weight on the new policy
theta_upd_wr   = 0.15 ; % Weight on the new policy
% options_vfi = optimoptions('fmincon','Display','none');
options_vfi = optimoptions('fminunc','Display','none');
%% Parameters
bbeta  = 0.97;
aalpha = 0.33;
phi    = 0.1 ;
xi     = 0.2 ;
psi    = 1.0 ;
delta  = 0.1 ;

par.bbeta = bbeta;
%% Steady State
tau_ss = 0.25;
z_ss   = 1.00;
           
% % RCE Steady State             
ce_steady_syst = @(x) [(1/x(1))-x(9)                                      ;
                    -x(2).^(psi)+x(9)*(1-tau_ss)*x(3)                    ;
                    x(3)-(1-aalpha)*z_ss*x(4).^(aalpha)*x(2).^(-aalpha)   ;
                    x(5)-tau_ss*x(3)*x(2)                                 ;
                    bbeta*(x(9)*(x(6))+(1-delta)*x(8))-x(8)               ;
                    x(9)-x(8)                                             ;
                    x(6)-(aalpha)*z_ss*x(4).^(aalpha-1)*x(2).^(1-aalpha)  ;
                    x(1)+x(5)+x(7)-z_ss*x(4).^(aalpha)*x(2).^(1-aalpha)   ;
                    delta*x(4)-x(7)                                       ;
                    ];
                
        
x0 = ones(9,1);
options = optimoptions('fsolve','Display','off');
[xce_sol,fval,exitflag,output] = fsolve(ce_steady_syst,x0,options);                

c_ss = xce_sol(1);
l_ss = xce_sol(2);
w_ss = xce_sol(3);
k_ss = xce_sol(4);
g_ss = xce_sol(5);
r_ss = xce_sol(6);
i_ss = xce_sol(7);
LMI_ss = xce_sol(8);
LMC_ss = xce_sol(9);

par.l_ss = l_ss;

%% Define Functions for VFI grid search 
fc   = @(i,w,r,fl,k,tau)(((1-tau).*w.*fl+r.*k-i));
fl   = @(c,w,tau)       (((1./c).*(1-tau).*w).^(1/(psi)));
fw   = @(fl,k,z,tau)    ((1-aalpha).*z.*(k.^(aalpha)).*fl.^(-aalpha));
fr   = @(fl,k,z,tau)    ((aalpha).*z.*(k.^(aalpha-1)).*fl.^(1-aalpha));
fg   = @(fl,fw,tau)     (tau*fl*fw);
faci = @(i,il)          (1-(phi/2)*((i./il)-1).^2);
fkp  = @(k,i,faci)      ((1-delta)*k + faci*i);
fu   = @(c,fg,fl)       ((log(c) + xi*log(fg) - (fl.^(1+psi))/(1+psi)));


%% State space: Exogenous States
% Taxes
tau_sd = 0.05;
vtau  = [tau_ss-tau_sd;tau_ss;tau_ss+tau_sd];
ntau = length(vtau)   ;
Ptau = [0.90 0.10 0.00;
        0.05 0.90 0.05;
        0.00 0.10 0.90];
% Productivity
lz = [-0.0673,-0.0336,0,0.0336,0.0673]';
vz = exp(lz)     ;
nz = length(vz)  ;
Pz = [0.9727 0.0273 0.0000 0.0000 0.0000;
      0.0041 0.9806 0.0153 0.0000 0.0000;
      0.0000 0.0082 0.9836 0.0082 0.0000;
      0.0000 0.0000 0.0153 0.9806 0.0041;
      0.0000 0.0000 0.0000 0.0273 0.9727];
 
 
% vtau = tau_ss;
% ntau = length(vtau)   ;
% vz = 1;
% nz = length(vz)  ;
% Pz = 1;
% Ptau = 1;

[zz,ttau] = ndgrid(vz,vtau);
vzz   = zz(:);
vttau = ttau(:); 

Pztau = kron(Ptau,Pz);  
%% State Space: Endogenous states
% Capital
nkp = 3;
vkp = linspace(0.7*k_ss,1.3*k_ss,nkp)';
% t-1 investment (lagged investment)
ni = 3;
vi = linspace(0.5*i_ss,1.5*i_ss,ni)';

[ii,kkp] = ndgrid(vi,vkp);
vii   = ii(:);
vkkp    = kkp(:); 

%% State space: Dimensions
nen = nkp*ni;
nex = ntau*nz;
n   = nex*nen;

%% 

auxvil = repmat(vi',nen,1)';
kp = ((1-delta)*vkkp + faci(auxvil',vii).*auxvil')';

mtau = repmat(vttau',nen,1);
mz   = repmat(vzz',nen,1);

%% Guesses
% Step 1. Define the initial guesses
% "Outter Loop"
mw_j = w_ss*ones(nen,nex);
mr_j = r_ss*ones(nen,nex);
% "Inner loop"
c_j    = c_ss*ones(nen,nex);
mu_j   = ones(nen,nex);
% Allocation
mup_j  = ones(nen,nex);
cp_j   = ones(nen,nex);
mrp    = ones(nen,nex);
aux1_acip = ones(nen,nex);
% Start the algorithm
for iter = 1:maxiter
% Step 2. Using guesses get labor, capital in t, and lagged i
% Labor
 l_j = (((1./c_j).*(1-mtau).*mw_j).^(1/(psi)));
% Capital with BC
 k_j =(c_j + vii - (1-mtau).*mw_j.*l_j)./mr_j;
% Lagged investment with LoM
 il_j = vii./(1+(((2/phi)*max((1-(vkkp-(1-delta).*k_j)),tol)).^(1/2)));
% Adjustment costs
 aci_j = faci(vii,il_j);
 aux1_aci = phi*((vii./il_j)-1).*(vii./il_j).^2;
 aux2_aci = 1-(phi*((vii./il_j)-1).*(vii./il_j) + (aci_j./vii));
% Step 3. Interpolate to find mup,aux1_acip
for iex = 1:nex
    mup_j(:,iex) = Bilinear_Interpolation(repmat(mu_j(:,iex),nen,1),[vii vkkp],il_j(:,iex),k_j(:,iex),1);
    cp_j(:,iex)  = Bilinear_Interpolation(repmat(c_j(:,iex),nen,1),[vii vkkp],il_j(:,iex),k_j(:,iex),1);
    mrp(:,iex)   = Bilinear_Interpolation(repmat(mr_j(:,iex),nen,1),[vii vkkp],il_j(:,iex),k_j(:,iex),1);
    aux1_acip(:,iex)   = Bilinear_Interpolation(repmat(aux1_aci(:,iex),nen,1),[vii vkkp],il_j(:,iex),k_j(:,iex),1);    
end
% Step 4. Update
aux_RHS = (mrp./cp_j) + (1-delta).*mup_j;
mu_jp = (bbeta*Pztau*aux_RHS')'; 

c_jp = 1./((bbeta*Pztau*(mup_j.*aux1_acip)')' + aux2_aci.*mu_j); c_jp = max(c_jp,tol);

% Step 5. 
 l_jp = (((1./c_jp).*(1-mtau).*mw_j).^(1/(psi)));
 k_jp =(c_jp + vii - (1-mtau).*mw_j.*l_jp)./mr_j; k_jp = max(k_jp,tol);    
%  mw_jp = ((1-aalpha).*mz.*(k_jp.^(aalpha)).*l_jp.^(-aalpha));
%  mr_jp = ((aalpha).*mz.*(k_jp.^(aalpha-1)).*l_jp.^(1-aalpha));
    
 aa = 1;   
    
end