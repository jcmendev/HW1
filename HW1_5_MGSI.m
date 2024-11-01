clc
clear all
close all
%% Description:
% This code solves by VFI with interpolation a Recursive Competitive Equilibrium of
% production economy with tax distortions, adjustment costs of investment
% It allows for doing multigrid algorithms, alternating between VF and PF iteration
% 
% 
% JCMV 2024
%% Technical Parameters 
maxiter        = 25000;
tol            = 1e-6 ;
theta_upd      = 0.2 ; % Weight on the new policy
theta_upd_wr   = 0.2 ; % Weight on the new policy
%% User Parameters
multigrid = 1; % Set 1 to do multigrid, 0 to solve VFI with 1 grid
    multg = [25 50 100 250]; % If =1, choose the multigrid
alternate = 1; % Set 1 to Alternate between VF and PF iteration
IRF_plots = 1; % Set 1 to do the IRFs
%% Parameters
bbeta  = 0.97;
aalpha = 0.33;
phi    = 0.1 ;
xi     = 0.2 ;
psi    = 1.0 ;
delta  = 0.1 ;

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


%% Define Functions for VFI grid search (This turned out to be inefficient)
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

[zz,ttau] = ndgrid(vz,vtau);
vzz   = zz(:);
vttau = ttau(:); 

Pztau = kron(Ptau,Pz);  

% vtau = tau_ss;
% ntau = length(vtau)   ;
% vz = 1;
% nz = length(vz)  ;
% Pz = 1;
% Ptau = 1;

% tic
%% State Space: Endogenous states
if multigrid 
    mgk = multg;
    nmgk = length(mgk);
else 
    mgk = 25;
    nmgk = length(mgk);
end
for ixmgk = 1:nmgk
    % Capital
    tic  
    nk = mgk(ixmgk);
    vk = linspace(0.2*k_ss,1.8*k_ss,nk)';
    % t-1 investment (lagged investment)
    nil = 20;
    vil = linspace(0.5*i_ss,1.5*i_ss,nil)';
    [iil,kk] = ndgrid(vil,vk);
    viil   = iil(:);
    vkk    = kk(:); 

    %% State space: Dimensions
    nen = nk*nil;
    nex = ntau*nz;
    n   = nex*nen;

    %% 

    auxvil = repmat(vil',nen,1)';
    kp = ((1-delta)*vkk + faci(auxvil',viil).*auxvil')';

    mtau = repmat(vttau',nen,1);
    mz   = repmat(vzz',nen,1);

    %% 
        %% 
        if ixmgk ==1
            mV   = zeros(nen,nex);
            policy = zeros(nen,nex);
            mr = r_ss*ones(nen,nex);
            mw = w_ss*ones(nen,nex);
            ml = l_ss*ones(nen,nex);
        else
            policy = zeros(nen,nex);
            mV   = Bilinear_Interpolation(RR.mV(:),[viil vkk],RR.vil,RR.vk,Pztau);
            mr   = Bilinear_Interpolation(RR.mr(:),[viil vkk],RR.vil,RR.vk,Pztau);
            mw   = Bilinear_Interpolation(RR.mw(:),[viil vkk],RR.vil,RR.vk,Pztau);
            ml   = Bilinear_Interpolation(RR.ml(:),[viil vkk],RR.vil,RR.vk,Pztau);
        end
      
    %         profile on
            fprintf('                VFI Interp Algorithm \n');
        if multigrid            
            fprintf('    Using Multigrid approach  by Chow and Tsitsiklis, (1991)\n');
        end
        if alternate
            fprintf('    Alternating between value and policy\n');
        end
        if ixmgk == 1
            fprintf('Using %d nodes for capital\n', mgk(ixmgk));
        else
            fprintf('Using %d nodes for capital, interpolating from %d nodes solution\n', mgk(ixmgk),mgk(ixmgk-1));              
        end
        fprintf('Iteration        Error\n');                 
    for iter = 1:maxiter
    % profile on 
% % Step 1 1. Guess prices, r_{j},w_{j}, labor, l_{j}, and value function \tilde{V}^{j}\left(k,i_{-},z,\tau;S\right) where j indexes the iteration.     
       mV_old = mV; 
       mw_old = mw;
       mr_old = mr;
       ml_old = ml;
% Step 2. For each node in the state space \left(k,i_{-},z,\tau\right), compute \sum_{z'\tau'|z,\tau}\pi\left(z',\tau|z,\tau\right)\tilde{V}^{j}\left(k',i,z',\tau';S'\right) by bilinearly interpolating the guess \tilde{V}^{j}\left(k,i_{-},z,\tau;S\right) on \left(k',i\right) since k' might not take values on the grid       
            Ev = (Pztau*mV')';   
            Ev_int_aux = Bilinear_Interpolation(Ev(:),[auxvil(:), kp(:)],vil,vk,Pztau);
            Ev_int = permute(reshape(Ev_int_aux',nex,nil,nen),[3 1 2]);
        ix = 1;
        for ixex = 1:nex
            z   = vzz(ixex)  ;
            tau = vttau(ixex);                                  
            for ixen = 1:nen
                k   = vkk(ixen)   ;                        
                il  = viil(ixen) ;  
                w   = mw_old(ixen,ixex);
                r   = mr_old(ixen,ixex);
                l   = ml_old(ixen,ixex);

% Step 3-5. Compute consumption c_{j} using the household budget constraint, guesses and government expenditure in steady statec_{j}=\left(1-\tau\right)w_{j}l_{j}+r_{j}k-i                
                c = (((1-tau).*w.*l+r.*k-vil')); c = max(c,0);
                g = (tau*l_ss*w) ;
                vl = (((1./c).*(1-tau).*w).^(1/(psi)));
                u = ((log(c) + xi*log(g) - (vl.^(1+psi))/(1+psi)));
                aux_Ev = Ev_int(ixen,ixex,:);            
% Step 6-8. Compute the objective function, F. 7. Linearly search the maximum of F, to find \tilde{V}^{j+1}\left(k,i_{-},z,\tau;S\right)             
                if alternate
                    if mod(iter,5)==0 || iter == 1
                        F = (1-bbeta)*u + bbeta*aux_Ev(:)';
                        [Vupd,xpol]       = max(F,[],2);       
                        mV(ixen,ixex)     = Vupd;  
                        policy(ixen,ixex) = xpol; 
                    else
                        aux_Ev = aux_Ev(:)';
                        F = (1-bbeta)*u(policy(ixen,ixex)) + bbeta*aux_Ev(policy(ixen,ixex));                        
                    end
                else
                    F = (1-bbeta)*u + bbeta*aux_Ev(:)';
                    [Vupd,xpol]       = max(F,[],2);       
                    mV(ixen,ixex)     = Vupd;  
                    policy(ixen,ixex) = xpol; 
                end
            end          
        end
% Step 8. With the policies compute consumption c_{j+1}, labor l_{j+1} for the next iteration
                mc  = fc(vil(policy),mw,mr,ml,vkk,mtau); mc = max(mc,0);
                ml = fl(mc,mw,mtau) ;
                mw = fw(ml,vkk,mz,mtau) ;
                mr = fr(ml,vkk,mz,mtau) ;            

% Step 9. Check convergence 
error_it  = max(abs([mV_old(:)-mV(:),mw_old(:)-mw(:),mr_old(:)-mr(:),ml_old(:)-ml(:)]));
        if mod(iter,50)==0 || iter == 1
    %         fprintf('%d ........... %.6f\n', iter,error_it);
            fprintf('%d ........... %.6f %.6f %.6f %.6f\n', iter,error_it(1),error_it(2),error_it(3),error_it(4));
        end
    % % Check convergence
        if max(error_it) < tol || any(isnan(error_it))
            break
        end
    % % Update for the next iteration
        mV = theta_upd*mV+(1-theta_upd)*mV_old;    
        mw = theta_upd_wr*mw+(1-theta_upd_wr)*mw_old;    
        mr = theta_upd_wr*mr+(1-theta_upd_wr)*mr_old;   
        ml = theta_upd_wr*ml+(1-theta_upd_wr)*ml_old;       
    %     profile off
    %     profile viewer
    end

    %% Store MG results 

        RR.vk   = vk;
        RR.vil  = vil;
        RR.mV   = mV;
        RR.mr   = mr;
        RR.mw   = mw;
        RR.ml   = ml;
fprintf('%d ........... %.6f %.6f %.6f %.6f\n', iter,error_it(1),error_it(2),error_it(3),error_it(4));
     toc   
end


%% Policies 
for ixen = 1:nen
    for ixex = 1:nex
        kpol(ixen,ixex) = kp(policy(ixen,ixex),ixen);
    end
end
kpol = reshape(kpol,nil,nk,nz,ntau); 
ipol = reshape(vil(policy),nil,nk,nz,ntau); 
cpol = reshape(mc,nil,nk,nz,ntau); 
Vpol = reshape(mV,nil,nk,nz,ntau); 
weq  = reshape(mw,nil,nk,nz,ntau); 
req  = reshape(mr,nil,nk,nz,ntau); 

[~,kpos] = min(abs(k_ss-vk));
[~,ipos] = min(abs(i_ss-vil));

vec = @(x) x(:);
figure()
subplot(1,2,1)
hold on
plot(vk,vk,'r')
plot(vk,vec(kpol(ipos,:,1,2)))
plot(vk,vec(kpol(ipos,:,end,2)))
plot(vk,vec(kpol(ipos,:,3,1)))
plot(vk,vec(kpol(ipos,:,3,3)))
xlim([min(vk) max(vk)])
ylim([min(vk) max(vk)])
xlabel('k','Interpreter','Latex')
xlabel('k''','Interpreter','Latex')
hold off
title('Capital Policy Function')
subplot(1,2,2)
hold on
plot(vil,vil,'r')
plot(vil,vec(ipol(:,kpos,1,2)))
plot(vil,vec(ipol(:,kpos,end,2)))
plot(vil,vec(ipol(:,kpos,3,1)))
plot(vil,vec(ipol(:,kpos,3,3)))
xlim([min(vil) max(vil)])
ylim([min(vil) max(vil)])
xlabel('i','Interpreter','Latex')
xlabel('i-','Interpreter','Latex')
hold off
title('Investment Policy Function')
print(['Fig1_Base'],'-depsc','-r0')
%%
if IRF_plots 
else
return
end
%% IRF to productivity shock
TT=100;
IR = zeros(nex,TT-1);
aux = zeros(nex,1); aux(6) = 1;
IR(:,1) = aux;
for t = 2:TT-1
    IR(:,t) = aux'*Pztau;
    aux = IR(:,t);
end
z_t = [1,vzz'*IR]';
k_t(1) = k_ss; [~,kpos] = min(abs(k_ss-vk));
i_t(1) = i_ss; [~,ipos] = min(abs(i_ss-vil));
c_t(1) = c_ss;

zpos   = 1;
taupos = 2;

t = 2;
k_t(t,1) = kpol(ipos,kpos,zpos,taupos); 
i_t(t,1) = ipol(ipos,kpos,zpos,taupos); 
[~,ipos] = min(abs(i_t(t,1)-vil));
c_t(t,1) = cpol(ipos,kpos,zpos,taupos); 

for t = 3:TT
    auxk = kpol(ipos,:,:,taupos);
    k_t(t,1)   = Bilinear_Interpolation(auxk(:),[k_t(t-1,1) z_t(t-1,1)],vk,vz,1);
    auxi = ipol(ipos,:,:,taupos);
    i_t(t,1)   = Bilinear_Interpolation(auxi(:),[k_t(t-1,1) z_t(t-1,1)],vk,vz,1);
    auxc = cpol(ipos,:,:,taupos);
    c_t(t,1)   = Bilinear_Interpolation(auxc(:),[k_t(t-1,1) z_t(t-1,1)],vk,vz,1);    
end

figure()
subplot(2,2,4)
plot(1:TT,i_t)
title('Investment')
subplot(2,2,3)
plot(1:TT,k_t)
title('Capital')
subplot(2,2,2)
plot(1:TT,c_t)
title('Consumption')
subplot(2,2,1)
plot(1:TT,z_t)
title('Productivity shock')
print(['Fig2_Base'],'-depsc','-r0')
%% IRF to Tax shock
TT=100;
IR = zeros(nex,TT-1);
aux = zeros(nex,1); aux(13) = 1;
IR(:,1) = aux;
for t = 2:TT-1
    IR(:,t) = aux'*Pztau;
    aux = IR(:,t);
end
tau_t = [1,vttau'*IR]';
k_t(1) = k_ss; [~,kpos] = min(abs(k_ss-vk));
i_t(1) = i_ss; [~,ipos] = min(abs(i_ss-vil));
c_t(1) = c_ss;

zpos   = 3;
taupos = 3;

t = 2;
k_t(t,1) = kpol(ipos,kpos,zpos,taupos); 
i_t(t,1) = ipol(ipos,kpos,zpos,taupos); 
[~,ipos] = min(abs(i_t(t,1)-vil));
c_t(t,1) = cpol(ipos,kpos,zpos,taupos); 

for t = 3:TT
    auxk = kpol(ipos,:,:,taupos);
    k_t(t,1)   = Bilinear_Interpolation(auxk(:),[k_t(t-1,1) z_t(t-1,1)],vk,vz,1);
    auxi = ipol(ipos,:,:,taupos);
    i_t(t,1)   = Bilinear_Interpolation(auxi(:),[k_t(t-1,1) z_t(t-1,1)],vk,vz,1);
    auxc = cpol(ipos,:,:,taupos);
    c_t(t,1)   = Bilinear_Interpolation(auxc(:),[k_t(t-1,1) z_t(t-1,1)],vk,vz,1);    
end

figure()
subplot(2,2,4)
plot(1:TT,i_t)
title('Investment')
subplot(2,2,3)
plot(1:TT,k_t)
title('Capital')
subplot(2,2,2)
plot(1:TT,c_t)
title('Consumption')
subplot(2,2,1)
plot(1:TT,tau_t)
title('Tax shock')
print(['Fig3_Base'],'-depsc','-r0')
