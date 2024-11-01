clc
clear all
close all
%% III. Optimization: basic problem 
% Parameters 
tol      = 1e-6 ;  % Tolerance 
maxiter  = 1000 ;  % Iterations
nvar     = 2    ;
self     = 0    ;  % 1 for self coded, 0 for built-in packages  
% Create the handle 
syms x y
f_sym      = 100*(y-x.^2).^2+(1-x).^2   ;
grad_f_sym = gradient(f_sym, [x, y])    ; 
H_f_sym    = hessian(f_sym, [x, y])     ;
% Convert symbolic expressions to function handles for numerical evaluation
f         = matlabFunction(f_sym);
pf        = matlabFunction(f_sym,'Vars', {[x,y]});
grad_f    = matlabFunction(grad_f_sym);
H_f       = matlabFunction(H_f_sym);

init_g = [2 2];
%% Minimizing
% Brute Force
% %{
tic 
[X_mat,Y_mat] = ndgrid([0:.01:2],[0:.01:2]);
Z_mat = f(X_mat,Y_mat);
Xvec = X_mat(:);
Yvec = Y_mat(:);
Zvec = Z_mat(:);
figure()
mesh(X_mat,Y_mat,f(X_mat,Y_mat))
[a,posmin] = min(Zvec);
BF_time = toc;
fprintf('    Brute Force  \n');
fprintf('Min of the function found: x = %.6f, y = %.6f\n', Xvec(posmin), Yvec(posmin));
fprintf('Solved in %2.2f secs\n', BF_time);
%}


%% Newton-Raphson 
% Initial guess for Newton-Raphson
x0 = init_g(1);  % Initial guess for x
y0 = init_g(2);  % Initial guess for y

fprintf('    Newton-Raphson  \n');
fprintf('Iteration        Error\n');
tic
for iter = 1:maxiter
    % Evaluate the function, gradient, and Hessian at current guess
    grad_F      = grad_f(x0, y0);
    H_inv = inv(H_f(x0, y0));
    % Update step using Newton-Raphson formula
    x_upd = [x0; y0] - H_inv*grad_F;    
    error_it = norm(x_upd - [x0; y0]);
    fprintf('%d ........... %.6f\n', iter,error_it);
    % Check convergence
    if error_it < tol
        break
    end   
    % Update x0 and y0 for the next iteration
    x0 = x_upd(1);
    y0 = x_upd(2);
end
NR_time = toc;
fprintf('Min of the function found: x = %.6f, y = %.6f, f(x,y)=%2.6f\n', x_upd(1), x_upd(2),f(x0, y0));
fprintf('Solved in %2.2f secs\n', NR_time);

methods_results(:,1) = x_upd;
%% BFGS
% Initial guess for bfgs
x0 = init_g(1);  % Initial guess for x
y0 = init_g(2);  % Initial guess for y

fprintf('    BFGS  \n');
if self
fprintf('Iteration        Error\n');
tic
B = eye(2);
for iter = 1:maxiter
    % Evaluate the function, gradient, and Hessian at current guess
    grad_F      = grad_f(x0, y0);
    search_dir  = -B*grad_F;
    % Update step using BFGS formula
    x_upd = [x0; y0] + search_dir;    
    % Update B
    S = x_upd-[x0;y0];
    Y = grad_f(x_upd(1), x_upd(2))-grad_F;
    B = (eye(2)-(S*Y')/(Y'*S))*B*(eye(2)-(Y*S')/(Y'*S)) + (S*S')/(Y'*S);    
    error_it = norm(x_upd - [x0; y0]);
    if mod(iter,10)==0 || iter == 1
    fprintf('%d ........... %.6f\n', iter,error_it);
    end
    % Check convergence
    if error_it < tol
        fprintf('%d ........... %.6f\n', iter,error_it);
        break
    end   
    % Update x0 and y0 for the next iteration
    x0 = x_upd(1);
    y0 = x_upd(2);
end
BFGS_time = toc;
else

options = optimoptions('fminunc', 'Algorithm', 'quasi-newton');
tic
[x_upd, fval] = fminunc(pf, init_g, options);
x_upd = x_upd';
BFGS_time = toc;
end
fprintf('Min of the function found: x = %.6f, y = %.6f, f(x,y)=%2.6f\n', x_upd(1), x_upd(2),f(x0, y0));
fprintf('Solved in %2.2f secs\n', NR_time);
methods_results(:,2) = x_upd;

%%
% % % Steepest descent
aalpha = 0.001; % 
maxiter  = 30000  ;  % Iterations

% Initial guess for Steepest descent
x0 = init_g(1);  % Initial guess for x
y0 = init_g(2);  % Initial guess for y

fprintf('    Steepest descent  \n');
fprintf('Iteration        Error\n');
tic
for iter = 1:maxiter    
    % Evaluate the function, gradient, and Hessian at current guess
    grad_F  = grad_f(x0, y0);  
    d = -grad_F/norm(grad_F);
    % Update step using formula
    x_upd = [x0; y0]  + aalpha*d;    
    error_it = norm(grad_F);
    if mod(iter,10000)==0 || iter == 1
    fprintf('%d ........... %.6f\n', iter,error_it);
    end
    % Check convergence
    if error_it < tol
        break
    end   
    % Update x0 and y0 for the next iteration
    x0 = x_upd(1);
    y0 = x_upd(2);
end
SD_time = toc;
fprintf('Min of the function found: x = %.6f, y = %.6f, f(x,y)=%2.6f\n', x_upd(1), x_upd(2),f(x0, y0));
fprintf('Solved in %2.2f secs\n', SD_time);
methods_results(:,3) = x_upd;
%%
% % % Conjugate descent
aalpha = 0.01; % 
maxiter  = 100000  ;  % Iterations

% Initial guess for Conjugate descent
x0 = init_g(1);  % Initial guess for x
y0 = init_g(2);  % Initial guess for y

fprintf('    Conjugate descent  \n');
if self
fprintf('Iteration        Error\n');
tic
fprintf('Iteration        Error\n');

grad_F  = grad_f(x0, y0);
d = -grad_F;
for iter = 1:maxiter        
        
    % Update step using Newton-Raphson formula
    x_upd      = [x0; y0]  + aalpha*d;   
    grad_F_new = grad_f(x_upd(1),x_upd(2));
    beta_it    = max((grad_F_new'*(grad_F_new))/(grad_F'*grad_F),0) ;   
    d = -grad_F_new/norm(grad_F_new) + beta_it*d;       
        
    error_it = norm(grad_F);
    if mod(iter,1)==0 || iter == 1
    fprintf('%d ........... %.6f\n', iter,error_it);
    end
    % Check convergence
    if error_it < tol || isnan(error_it)
        break
    end
    % Update x0 and y0 for the next iteration
    x0 = x_upd(1);
    y0 = x_upd(2);
    grad_F = grad_F_new;
end
else
tic

%% 
% Set up iterative conjugate gradient using pcg for approximate minimization
xk = init_g;  % Initial point
% grad_f = gradient(f_sym, [x, y]);
% grad_f_func = matlabFunction(grad_f, 'Vars', {x, y});


for k = 1:maxiter
    % Calculate the gradient at the current point
    grad_eval = grad_f(xk(1), xk(2));
    
    % Approximate Hessian using Jacobian of the gradient (linear approximation)
    H_approx = jacobian(grad_f, [x, y]);
    H_approx_func = matlabFunction(H_approx, 'Vars', {x, y});
    Hk = H_approx_func(xk(1), xk(2));
    
    % Run pcg on the approximate Hessian system
    [dx, flag] = pcg(Hk, -grad_eval, tol, maxiter);
    
    % Update x
    xk = xk + dx;

    % Check convergence (stop if gradient norm is below tolerance)
    if norm(grad_eval) < tol
        break;
    end
end

disp('Minimizing point:');
disp(xk);
disp('Objective function value at minimum:');
disp(double(subs(f_sym, {x, y}, {xk(1), xk(2)})));

x_upd = xk(1,:)';
CD_time = toc;
end
fprintf('Min of the function found: x = %.6f, y = %.6f, f(x,y)=%2.6f\n', x_upd(1), x_upd(2),f(x_upd(1),x_upd(2)));
fprintf('Solved in %2.2f secs\n', CD_time);
methods_results(:,4) = x_upd;


