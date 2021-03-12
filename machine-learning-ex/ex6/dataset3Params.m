function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
C_opt = 1;
sigma = 0.3;
sig_opt = 1;
steps = [0.01 0.03 0.1 0.3 1 3 10 30 90];

err_opt = 1000;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for C_ind = 1:length(steps)

    for sig_ind = 1:length(steps)

        model = svmTrain(X, y, steps(C_ind), @(x1, x2)gaussianKernel(x1, x2, steps(sig_ind)));

        prediction = svmPredict(model, Xval);
    
        if err_opt > mean(double(prediction ~= yval))
            err_opt = mean(double(prediction ~= yval));
            C_opt = C_ind;
            sig_opt = sig_ind;
        end

    end

end

C = steps(C_opt);
sigma = steps(sig_opt);

% =========================================================================

end
