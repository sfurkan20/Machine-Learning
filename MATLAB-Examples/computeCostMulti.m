function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

    m = length(y); % number of training examples

    function retVal = h(x)    % x: variable vector
        retVal = dot(x, theta);
    end

    rmsSum = 0;
    for i = 1:m
        rmsSum = rmsSum + (h(X(i, :)) - y(i, 1)) ^ 2;
    end

    J = rmsSum / (2 * m);
end
