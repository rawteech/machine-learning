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
sigma = 0.3;

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

% possible ranges of C and sigma
C_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
Sigma_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% optimal ranges of C and sigma
optimal_C = C_range(1);
optimal_sigma = Sigma_range(1);

% previous error
previous_error = 1000;

% now to get optimal C and Sigma we loop through
for i=1:length(C_range)
    
    now_c = C_range(i);
    
    for j=1:length(Sigma_range)
        
        currentSigma = Sigma_range(j);
        model = svmTrain(X, y, now_c, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        
        if err < previous_error

            optimal_C = now_c;
            optimal_sigma = currentSigma;
            previous_error = err;
        end;
    end;
end;

% our final returned values required
C = optimal_C;
sigma = optimal_sigma;


% =========================================================================

end
