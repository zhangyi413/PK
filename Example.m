clc
clear

load('branin_2d_10.mat'); % training samples Branin-Hoo function
load('branin_2d_5000_te.mat'); % testing samples Branin-Hoo function

% load('Rosenbrock_12d_120.mat'); % training samples Rosenbrock function
% load('Rosenbrock_12d_10000_te.mat'); % testing samples Rosenbrock function

X = trX{1}; % training samples X 
Y = trY{1}; % training samples Y
Xtest = teX{1}; % testing samples X
Ytest = teY{1}; % testing samples Y

lb = 0.1*ones(1, size(trX{1}, 2)); % The lower bound of the correlation parameters
ub = 5*ones(1, size(trX{1}, 2)); % The upper bound of the correlation parameters
theta0 = ones(1, size(trX{1}, 2)); % The initial values of the correlation parameters

numRegu = 17; % The number of candidates of the regularization parameter for cross-validation
 
kfolds = 10; % The number of folds for cross-validation
idxFolds = randperm(size(X, 1)); % Training samples random allocation for cross-validation

[ Yhat ] = LassoKriging( X, Y, Xtest, theta0, lb, ub, numRegu, kfolds, idxFolds); % Fitting the model

errors = Yhat - Ytest;
rmse = sqrt( mean( (errors.^2) ) );
rrmse = sqrt( mean( (errors.^2) ) )/std(Ytest, 1);