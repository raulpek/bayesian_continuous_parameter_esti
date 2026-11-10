%% Task 1
clc
clearvars
close all
% 1 Consider a GP regression problem

%{ 
1.1) Simulate the problem and measurements
starting from a noisy Matérn realization
and plot all relevant quantities, such as
the measurements and the true unknown
%}

h = 0.02;
L_dom = 10;
N = round(L_dom/h) +1;
t1 = (0:N-1)'*h;
R_dist_N = abs(t1-t1');
% parameters
nu = 1.5;
sigma_l_true = 0.40;
sigma_f = 1.0;
sigma_n = 0.10;
% R_dist, sigma_l, nu, sigma_f
rng(42);
X_true = matern_own(R_dist_N,sigma_l_true,nu,sigma_f);
% Ground Truth
obs_idx = (1:5:N);
y_obs = X_true(obs_idx) + sigma_n*randn(numel(obs_idx),1);

H = sparse(1:numel(obs_idx), obs_idx, 1,numel(obs_idx), N);
%{ 
1.2) Estimate the prior correlation length ℓ 
and all the states. Choose the prior for ℓ 
to be either uniform or log-Gaussian. Use the
full covariance matrix and the stochastic 
partial differential equation (SPDE) 
approximation for faster computations. 
What is the computational gain between 
these two models?
%}

% extended domain
extend = round(2*sigma_l_true/h);
Ne = N + 2*extend;
inner = extend + (1:N);
e_e = ones(Ne,1);
% t1_e = ((-extend):(N-1+extend))'*h;
H_e = sparse(1:numel(obs_idx), obs_idx+extend, 1,...
    numel(obs_idx), Ne);
cos_vec_N = cos(pi*(1:N)' /(N+1));
cos_vec_Ne = cos(pi*(1:Ne)' /(Ne+1));

% spde calibration at true_sigma_l
kappa_curr = sqrt(3) / sigma_l_true;
a0_curr = (kappa_curr^2) + (2 / h^2);
[L0,~] = get_diric_L(a0_curr, h, N, cos_vec_N);
e_mid = zeros(N,1);
e_mid(round(N/2)) = 1;
std_center0 = norm(L0 \ e_mid); % symmetry
c_scale = sigma_f / (std_center0*kappa_curr^(3/2));

%{ 
1.5) Use the preconditioned Crank–Nicolson (pCN)
algorithm for MCMC sampling. Plot the chains,
and consider the effect of burn-in, chain
length, and thinning.
%}
n_mcmc = 20000;
burnIn = 3000;
ini_sig = 0.40;
thin_fac = 10;
beta = 0.05;
tuning = 0.15;
% running the algorithm
methods = ["full_cov", "spde", "spde_ext"];
res = struct();

time_cpu = zeros(1,3);
for mi = 1:numel(methods)
    method = methods(mi);
    if method == "spde_ext"
        H_m = H_e;
        N_m = Ne;
        cos_m = cos_vec_Ne;
        R_m = [];
    else
        H_m = H;
        N_m = N;
        cos_m = cos_vec_N;
        R_m = R_dist_N;
    end
    rng(mi);
    tic;
    res.(method) = pcn_algo(H_m, y_obs, N_m,...
        h, sigma_f, sigma_n, nu, ini_sig,...
        method, n_mcmc, burnIn, thin_fac,...
        beta, tuning, R_m, ...
        cos_m, c_scale);
    time_cpu(mi) = toc;
end
% res_all = {res.full_cov, res.spde, res.spde_ext};

% check results
fprintf("Gain full_cov/SPDE: %.1fx\n", time_cpu(1)/time_cpu(2));
result_summary(res, methods, X_true, t1, t1(obs_idx), y_obs,...
    inner,thin_fac, burnIn, sigma_l_true, "Task1", 1);

%% Functions

% box car function
function K = box_car_fn(h, w, half_w, N, extend)
    if N(2) == 0
        K = zeros(N,N);
        for i = 1:N
            for j = 1:N
                if abs((i-j)*h) <= half_w
                    K(i,j) = 1/w *h;
                end
            end
        end
    else
        K = zeros(N(1),N(2));
        for i = 1:N(1)
            for j = 1:N(2)
                if abs(((i+extend)-j)*h) <= half_w
                    K(i,j) = 1/w *h;
                end
            end
        end

    end


end

% Matern Covariance
function X = matern_own(R_dist, sigma_l, nu, sigma_f)
    N = size(R_dist,1);
    C = matern_cov(R_dist, sigma_l, nu, sigma_f);
    X = chol(C + 1e-6*eye(N), 'lower')*randn(N,1);
end

function K = matern_cov(R_dist, sigma_l, nu, sigma_f)
    if nu == 1.5
        arg = (sqrt(3)/sigma_l)*R_dist;
        K = sigma_f^2 * (1 + arg) .* exp(-arg);
    else
        r = R_dist;
        r(r==0) = eps;
        arg = (sqrt(2*nu)*r/sigma_l);
        K = sigma_f^2 *(2^(1-nu)/gamma(nu))*(arg.^nu) .*...
            besselk(nu,arg);
        K(logical(eye(size(K)))) = sigma_f^2;
    end
end


% Prior distribution
function lp = log_prior(sigma)
    m = log(0.5);
    s = 0.7;
    if sigma > 0
        lp = -log(sigma*s*sqrt(2*pi)) - 0.5*...
            ((log(sigma)-m)/s)^2;
    else
        lp = -Inf;
    end
end


% to calculate phi
function phi = phi_calc(x, y_obs, sigma_n, H)
    res = y_obs - H*x;
    phi = 0.5*sum(res.^2) / sigma_n^2;
end

function [L, logdetL] = get_diric_L(a0, h, N, cos_vec)
    e = ones(N,1);
    a1 = -1/h^2;
    L = spdiags([a1*e, a0*e, a1*e], [-1 0 1], N, N);
    lam = a0 + 2*a1*cos_vec;
    logdetL = sum(log(lam));
end

function [a0,scale] = spde_params_calib(sigma_l,...
    h, c_scale)
    kappa = sqrt(3) / sigma_l;
    a0 = kappa^2 + 2/h^2;
    scale = c_scale *kappa^(3/2);
end


% PCN1
function res = pcn_algo(H, y_obs, N,...
    h, sigma_f, sigma_n, nu, ini_sig, method, n_mcmc,...
    burnIn, thin_fac, beta, tuning, R_dist,...
    cos_vec, c_scale)
    accepted = 0;
    acc_sigma_l = 0;
    sigma_l = ini_sig;

    % thinning allocation
    n_saved = floor(n_mcmc / thin_fac );
    samples_sigma_l = zeros(1, n_saved);
    samples_u =  zeros(N, n_saved);
    saves_i = 0;

    if method == "full_cov"
        C = matern_cov(R_dist, sigma_l, nu, sigma_f);
        L_curr = chol(C + 1e-8*eye(N), 'lower');
        logdet = sum(log(diag(L_curr)));
        scale = 1;
    else
        [a0, scale] = spde_params_calib(sigma_l,...
            h, c_scale);
        [L_curr, logdet] = get_diric_L(a0, h, N, cos_vec);

    end

    u_curr = zeros(N,1);
    for i = 1:n_mcmc
        % proposal of latent WN u
        z = randn(N,1);
        if method == "full_cov"
            xi = L_curr*z;
        else
            xi = L_curr \ (scale*z);
        end
        u_prop = sqrt(1-beta^2)*u_curr + beta*xi;
        % acceptance rate to u
        lg_a_u = -phi_calc(u_prop,y_obs,sigma_n, H)...
                + phi_calc(u_curr,y_obs,sigma_n,H);
        if log(rand()) < lg_a_u
            u_next = u_prop;
            accepted = accepted + 1;
        else
            % reject
            u_next = u_curr;
        end
        % proposal sigma_l
        lg_sigma_l = log(sigma_l);
        lg_sigma_l_prop = lg_sigma_l + tuning *randn();
        sigma_l_prop = exp(lg_sigma_l_prop);
        % prior sigma_l
        lg_prior_curr = log_prior(sigma_l);
        lg_prior_prop = log_prior(sigma_l_prop);

        if isfinite(lg_prior_prop)
            if method == "spde" || method == "spde_ext"
                Lu_c = L_curr*u_next;
                log_p_curr = logdet - N*log(scale) - 0.5*...
                    (Lu_c'*Lu_c) / scale^2;
                [a0_prop, scp] = spde_params_calib(sigma_l_prop,...
                    h, c_scale);
                [L_prop,log_det_prop] = get_diric_L(a0_prop, h, N, cos_vec);
                Lu_prop = L_prop *u_next;
                log_p_prop = log_det_prop - N*log(scp) -0.5*...
                    (Lu_prop' *Lu_prop) / scp^2;
            else
                % current field prior dens.
                v_curr = L_curr \ u_next;
                log_p_curr = -logdet - 0.5*(v_curr'*v_curr);
                % proposed field prior dens.
                C_prop = matern_cov(R_dist, sigma_l_prop,...
                    nu, sigma_f);
                L_prop = chol(C_prop + 1e-8*eye(N), 'lower');
                % field prior
                log_det_prop = sum(log(diag(L_prop)));
                v_prop = L_prop \ u_next;
                log_p_prop = -log_det_prop -...
                    0.5*(v_prop' *v_prop);
            end
            % metropolis hastings
            lp_a = (log_p_prop - log_p_curr) + ...
                (lg_prior_prop - lg_prior_curr) + ...
                (lg_sigma_l_prop - lg_sigma_l);

            if log(rand()) < lp_a
                sigma_l = sigma_l_prop;
                L_curr = L_prop;
                logdet = log_det_prop;
                if method == "spde" || method == "spde_ext"
                    scale = scp;
                end
                acc_sigma_l = acc_sigma_l + 1;
            end
        end
        u_curr = u_next;
        % sample thinning
        if mod(i, thin_fac) == 0
            saves_i = saves_i + 1;
            samples_u(:,saves_i) = u_curr;
            samples_sigma_l(saves_i) = sigma_l;
        end
    end
    % sample processing
    burnIn_thin = floor(burnIn / thin_fac);
    
    X_post = samples_u(:, (burnIn_thin+1):end);
    sigma_post = samples_sigma_l(:, (burnIn_thin+1):end);
    
    mu_X = mean(X_post,2);
    std_X= std(X_post,0,2);

    acc_rate = accepted / (n_mcmc-1)*100;
    acc_rate_sigma_l = acc_sigma_l /(n_mcmc-1)*100;
    % create output
    res = struct( ...
    "results", X_post,...
    "mean", mu_X,...
    "std", std_X,...
    "acceptance_rate", acc_rate,...
    "sigma_samp", sigma_post,...
    "full_u", samples_u, ...
    "full_sigma", samples_sigma_l,...
    "sigma_acceptance_rate",acc_rate_sigma_l);
end

function result_summary(res, methods, x_true, t, t_y, y,...
    inner,thin_fac, burnIn, l_ref, tag, fig0)
    N = numel(t);
    nm = numel(methods);
    mid = round(N/2);
    nodes = [1 mid N];
    xm = cell(1,nm);
    low = cell(1,nm);
    up = cell(1,nm);
    S = cell(1,nm);
    rmse = zeros(nm,1);
    cover = zeros(nm,1);
    El = zeros(nm,1);
    rows = cell(1,nm);

    % iterate through the models
    for m = 1:nm
        r = res.(methods(m));
        rows{m} = 1:N;
        if methods(m) == "spde_ext"
            rows{m} = inner;
        end
        S{m} = r.results(rows{m}, :);
        xm{m} = mean(S{m},2);
        low{m} = prctile(S{m}, 2.5, 2);
        up{m} = prctile(S{m}, 97.5,2);
        rmse(m) = sqrt(mean((xm{m} - x_true).^2));
        cover(m) = mean(x_true >= low{m} & ...
            x_true <= up{m});
        El(m) = mean(r.sigma_samp);
    end
    % result table
    disp(table(methods(:), rmse, cover, El,...
        'VariableNames', {'Method', 'RMSE', 'Coverage', 'E_l'}));
    % chains
    figure(fig0);
    clf;
    for m = 1:nm
        r = res.(methods(m));
        subplot(2,nm,m);
        plot((1:numel(r.full_sigma))*thin_fac, r.full_sigma);
        hold on;
        xline(burnIn, 'r--');
        if ~isnan(l_ref)
            yline(l_ref,'k--');
        end
        title(tag + ": chain of l, "+ methods(m));
        xlabel("iteration");
        grid on;

        subplot(2,nm,nm+m);
        plot((1:size(r.full_u, 2))*thin_fac, r.full_u(rows{m}(mid),:));
        xline(burnIn, 'r--');
        yline(x_true(mid),'k--');
        title("stata chain at  t= " + t(mid) + ", "+ methods(m));

    end
    % chain length
    figure(fig0+1);
    clf;
    for m = 1:nm
        r = res.(methods(m));
        post = r.sigma_samp;
        subplot(2,nm,m);
        plot(cumsum(post) ./ (1:numel(post)));
        grid on;
        title(tag + "running mean of l, "+ methods(m));
        xlabel("saved sample");

        subplot(2,nm,nm+m);
        plot(0:50, simple_acf(post,50));
        hold on;
        plot(0:50, simple_acf(post(1:5:end),50));
        legend("as saved", "thinner by +5");
        grid on;
        title("ACF of l," + methods(m));
        xlabel("lag");
    end
    % marginal of l
    figure(fig0+2);
    clf;
    for m = 1:nm
        r = res.(methods(m));
        subplot(1,nm,m);
        histogram(r.sigma_samp, 40, 'Normalization','pdf');
        hold on;
        if ~isnan(l_ref)
            xline(l_ref, 'k--', 'LineWidth', 1.2);
        end
        title(tag + ": l marginal, " + methods(m));
        grid on;
    end
    % marginal of states at three nodes
    figure(fig0+3);
    clf;
    for m = 1:nm
        for q = 1:3
            subplot(nm,3,3*(m-1)+q);
            histogram(S{m}(nodes(q),:), 30, 'Normalization','pdf');
            hold on;
            xline(x_true(nodes(q)), 'k--', 'LineWidth', 1.2);
            title(sprintf("%s, t = %.1f",char(methods(m)), t(nodes(q))));
            grid on;
        end
    end
    % 95 perc. CI
    figure(fig0+4);
    clf;
    for m = 1:nm
        subplot(nm,1,m);
        fill([t; flipud(t)], [low{m}; flipud(up{m})], ...
            [0.85 0.9 1], 'EdgeColor', 'none');
        hold on;
        plot(t, xm{m}, 'b', 'LineWidth', 1.2);
        plot(t, x_true, 'k--');
        plot(t_y, y, 'ro', 'MarkerSize', 4);
        title(tag + ": 95% credible interval, " + methods(m));
        xlabel("t");
        grid on;
    end
    legend("95% CI", "True", "Posterior mean", "Data");
    % RMSE
    figure(fig0+5);
    clf;
    subplot(1,2,1);
    bar(rmse);
    set(gca, 'XTick', 1:nm, 'XTickLabel', methods);
    ylabel("RMSE");
    title(tag + ": RMSE by method");
    grid on;

    subplot(1,2,2);
    hold on;
    for m = 1:nm
        plot(t, abs(xm{m} -x_true));
    end
    legend(methods);
    xlabel("t");
    ylabel("|posterior mean - truth|");
    title(tag + ": point-wise error");
    grid on;

end

function a = simple_acf(x, maxlag)
    x = x(:) - mean(x);
    c = zeros(maxlag+1,1);
    for k = 0:maxlag
        c(k+1) = sum(x(1:end-k) .* x(1+k:end));
    end
    a = c / c(1);
end
