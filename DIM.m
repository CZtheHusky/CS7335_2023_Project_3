function n2 = L2_distance_2(x,c,df)
%   D2  = L2_distance_2(a, b, df)
%
%   Get the square of L2_distance of two sets of samples. It is useful when
%   you only need square of L2_distance, for example, in Guassian Kernel.
%   This function is faster and lower memory requirement compared with the
%   funtion L2_distance by Roland Bunschoten et al.
%
%   Input:
%       a   -- d-by-m matrix, i.e. m samples of d-dimension.
%       b   -- d-by-n matrix;
%       df  -- df = 1, force diagnal to zero, otherwise not.
%
%   Output:
%       D2  -- m-by-n matrix, the square of L2_distance.
%
%   Code from CHEN Lin, comment by LI Wen.
%

if nargin < 3
    df = 0;
end
[dimx, ndata] = size(x);
[dimc, ncentres] = size(c);
if dimx ~= dimc
	error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2), 1))' + ...
  		ones(ndata, 1) * sum((c.^2),1) - ...
  		2.*(x'*(c));
% make sure result is all real
n2 = real(full(n2));
n2(n2<0) = 0;
% force 0 on the diagonal?
if (df==1)
  n2 = n2.*(1-eye(size(n2)));
end

function [ap] = calc_ap(gt, desc, k)
%   ap = calc_ap_k(gt, desc, k)
%
%   Calculate the "average precision" of top-k elems in a ranking result.
%   Modified from the function "calc_ap".
%
%   Input:
%       gt      -- ground truth lables
%       desc    -- decision values
%       k       -- top k, optional. absent or assign a value less than one,
%                  will get ap.
%   Output:
%       ap@k    -- Average Precision calculated on top-k elems.
%
% by LI Wen

gt = gt(:);
desc = desc(:);
[dv, ind] = sort(-desc); dv = -dv;

if(exist('k', 'var') && k>0 && length(ind) > k)
    ind = ind(1:k);
end

gt = gt(ind);
pos_ind = find( gt > 0 );
npos = length(pos_ind);
if npos == 0
    ap = 0;
else
    ap = mean( (1:npos)' ./ pos_ind(:) );
end

function [ProjectMatrix, EigenValues] = calc_pca(features)
% [ProjectMatrix EigenValues] = perform_pca()
%
% Compute the PCA projection matrix, support sparse matrix for fast
%
% Input:
%   features : d-by-n matrix
% Output:
%   ProjectMatrix, EigenValues
%
% by LI Wen on 25 Sep, 2012
%

tic;
[dim, N]    = size(features);
mean_feat   = mean(features, 2);
features    = features - repmat(mean_feat, 1, N);
% after sub mean, it is not sparse, so we full it to parallelize
features    = full(features);
tt = toc;
fprintf('\tPCA:feature preprocssing time = %f\n', tt);

if dim <= N
    fprintf('\tDim < N, do cov decomposition\n');
    tic;
    cov = features*features';
    cov = full(cov);
    tt = toc;
    fprintf('\tPCA:cov computing time = %f\n', tt);
    tic;
    [eigVec eigVal] = eig(cov);
    tt = toc;
    fprintf('\tPCA:eig computing time = %f\n', tt);
else
    fprintf('\tDim > N, do kernel decomposition\n');
    tic;
    cov = features'*features;
    cov = full(cov);
    tt = toc;
    fprintf('\tPCA:cov computing time = %f\n', tt);
    tic
    [eigVec eigVal] = eig(cov);
    eigVec = features*eigVec;
    eigVec = eigVec ./ repmat(sqrt(sum(eigVec.^2)), [dim, 1]);    %normalize
    tt = toc;
    fprintf('\tPCA:eig computing time = %f\n', tt);
end
[EigenValues ind]   = sort(diag(eigVal), 'descend');
ProjectMatrix       = eigVec(:, ind);

function [kernel param] = getKernel(featuresA, featuresB, param)
% compute a kernel, it can be K(A, A) or K(A, B)
% Usage:
%  1. Compute the kernel between different examples, e.g. in testing:
%   [kernel param] = getKernel(featuresA, featuresB, param)
%  2. Compute the kernel between the sample exaples, e.g. in training:
%   [kernel param] = getKernel(features, param)
%
% Input:
%   featuresA: d-by-m matrix, d is feature dimension, m is the number of
%   samples
%   featuresB: d-by-n matrix, d is feature dimension, m is the number of
%   samples
%   param:  -kernel_type:
%               'linear', 'gaussian'
%           -(gaussian)ratio, sigma, gamma
%
% Output:
%   kernel: m-by-n or m-by-m matrix
%   param: depends on the kernel type
%
%  by LI Wen on Jan 04, 2012
%

if (nargin < 2)
    error('Not enough inputs!\n');
elseif (nargin < 3)
    param = featuresB;
    featuresB = featuresA;
end

if(~isfield(param, 'kernel_type'))
    error('Please specify the kernel_type!\n');
end

kernel  = [];
kt      = lower(param.kernel_type);
if(strcmp(kt, 'linear'))
    kernel = return_LinearKernel(featuresA, featuresB);
elseif(strcmp(kt, 'gaussian'))
    [kernel param] = return_GaussianKernel(featuresA, featuresB, param);
else
    error('Unknown type of kernel: %s.\n', param.kernel_type);
end

function [K, param] = return_GaussianKernel(featuresA, featuresB, param)

[dA nA] = size(featuresA);
[dB nB] = size(featuresB);

assert(dA == dB);

sq_dist = L2_distance_2(featuresA, featuresB);

if(~isfield(param, 'ratio') || param.ratio == 0)
    param.ratio = 1;
end

if(~isfield(param, 'gamma') || param.gamma == 0)
    if (~isfield(param, 'sigma') || param.sigma == 0)
        % use default sigma
        tmp = mean(mean(sq_dist))*0.5;
        param.sigma = sqrt(tmp);
    end
    % compute gamma according to param.ratio and param.sigma
    if(param.sigma == 0)
        param.gamma     = 0;
    else
        param.gamma     = 1/(2*param.ratio*param.sigma^2);
    end
else
    % already specify gamma, then sigma and ratio set to 0.
    if(~isfield(param, 'sigma'))
        param.sigma = 0;
    end
    if(~isfield(param, 'ratio'))
        param.ratio = 0;
    end
end

K = exp(-sq_dist*param.gamma);

function [K, param] = return_LinearKernel(featuresA, featuresB, param)

[dA nA] = size(featuresA);
[dB nB] = size(featuresB);

assert(dA == dB);

K = featuresA'*featuresB;

%#####
% for linear kernel, the features usually be sparse, so the K is also
% sparse matrix(but it usually not sparse). We need to full it, otherwise,
% the following operator on K maybe very slow.
%
% I don't know how about the non-linear case, should I move this to
% getKernel??
%
if(issparse(K))
    K = full(K);
end

function Y = solve_cg(Yk)

[n, p] = size(Yk);
FYk = dF(Yk);
Gk = FYk - Yk*Yk'*FYk;
Hk  = -Gk;
fprintf('#iter  |   obj\n');
for  k = 1 : 20
    [U,S, V] = svd(Hk, 'econ');
    theta = diag(S);

    [tk, obj(k)] = line_search(Yk, V, U, theta);

    Yk1 = Yk*V*diag(cos(theta*tk))*V' + U*diag(sin(theta*tk))*V';
    FYk1 = dF(Yk1);
    Gk1 = FYk1 - Yk1*Yk1'*FYk1;
    tauHk = (-Yk*V*diag(sin(theta*tk)) + U*diag(cos(theta*tk)))*diag(theta)*V';
    tauGk = Gk - (Yk*V*diag(sin(theta*tk)) + U*diag(1 - cos(theta*tk)))*U'*Gk;
    gammak = trace( (Gk1-tauGk)'*Gk1 ) / trace(Gk'*Gk);
    Hk1 = -Gk1 + gammak*tauHk;

    if mod(k+1, p*(n-p)) == 0
        Hk1 = -Gk1;
    end

    if k>1 && (abs((obj(k-1)-obj(k))/obj(k)) < 5e-4 ||obj(k)>obj(k-1))
        break;
    end

    fprintf('%d\t|\t%g\n', k, obj(k));

    Yk = Yk1;
    Hk = Hk1;
    Gk = Gk1;
end
Y = Yk;
end

function [t, f] = line_search(Y, V, U, ds)
    % linear search for t
    r = 0.5*(sqrt(5)-1);
    xs = 0;
    xe = 1;
    d  = (xe - xs)*r;
    x1 = xe - d;
    x2 = xs + d;
    f1 = func_obj(x1, Y, V, U, ds);
    f2 = func_obj(x2, Y, V, U, ds);
    tau = 0.001;
    while(1)
        d = d*r;
        if(f1<f2)
            xe = x2;

            x2 = x1;
            f2 = f1;

            x1 = xe -d;
            f1 = func_obj(x1, Y, V, U, ds);
        else
            xs = x1;

            x1 = x2;
            f1 = f2;

            x2 = xs + d;
            f2 = func_obj(x2, Y, V, U, ds);
        end
        if(abs(x1 - x2) < tau)
            break;
        end
    end
    if(f1 > f2)
        f = f1;
        t = x1;
    else
        f = f2;
        t = x2;
    end

end

function obj = func_obj(t, Y, V, U, theta)

    Yt = Y*V*diag(cos(theta*t))*V' + U*diag(sin(theta*t))*V';
    obj = F(Yt);

end

function W = trainDIP_CG(y, Xs, Xt, sigma, lambda, d)

D = size(Xs,2);

global FParameters
FParameters = [];
FParameters.Xs = Xs;
FParameters.Xt = Xt;
FParameters.D = D;
FParameters.d = d;
FParameters.sigma = sigma;
FParameters.ys = y;
FParameters.lambda = lambda;

Y = eye(D);
Y = Y(:,1:d);

global SGParameters;
SGParameters = [];
SGParameters.verbose = 1;
SGParameters.Mode = 0;
SGParameters.metric = 1;
SGParameters.complex = 0;
SGParameters.motion = 1;
SGParameters.gradtol = 1e-7;
SGParameters.ftol = 1e-10;
SGParameters.partition = num2cell(1:size(Y,2));
SGParameters.dimension = dimension(Y);


W = solve_cg(Y);

end



clear all;clc;

addpath('.\utils');
addpath('.\sg_min2.4.3');
addpath('.\tools\libsvm-3.17\matlab');

% parameter
param.lambda = 0;
param.dim = 200;
param.C = 1;

fprintf('loading data....\n');
train_data = load('.\data\train_data');
test_data = load('.\data\test_data');


Xs = train_data.train_features';
Xu = test_data.test_features';

PP = calc_pca([Xs; Xu]');
Xs = Xs * PP;
Xu = Xu * PP;

sigma = sqrt(0.5/calc_g(Xs));
sigma = 2*sigma^2;

W = trainDIP_CG(train_data.train_labels, Xs, Xu, sigma, param.lambda, param.dim);

train_feature = Xs * W;
test_feature = Xu * W;
clear W;

kparam = struct();
kparam.kernel_type = 'gaussian';
[K, kernel_param] = getKernel(train_feature', kparam);
test_kernel = getKernel(test_feature', train_feature', kernel_param);

train_kernel    = [(1:size(K, 1))' K];
para   = sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.C,0,4,1);
model  = svmtrain(train_data.train_labels, train_kernel, para);

ay      = full(model.sv_coef)*model.Label(1);
idx     = full(model.SVs);
b       = -(model.rho*model.Label(1));

decs    = test_kernel(:, idx)*ay + b;
ap  = calc_ap(test_data.test_labels, decs);


