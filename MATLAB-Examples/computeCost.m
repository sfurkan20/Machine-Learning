function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

    m = length(y); % number of training examples

    function retVal = h(x)
        retVal = theta(1, 1) + x * theta(2, 1);
    end

    rmsSum = 0;
    for i = 1:m
        rmsSum = rmsSum + (h(X(i, 2)) - y(i, 1)) ^ 2;
    end

    J = rmsSum / (2 * m);
end
