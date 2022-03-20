function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

    m = length(y); % number of training examples
    newThetas = theta;
    J_history = zeros(num_iters, 1); % stores the costs

    function retVal = h(x)    % x: variable vector
        retVal = dot(x, theta);
    end

    function retVal = J(th)    %th: thetas vector
        retVal = 0;
        for i = 1:m
            retVal = retVal + (h(X(i, :)) - y(i, 1)) ^ 2;
        end
        retVal = retVal / (2 * m);
    end

    function retVal = gradientValue(thetaIndex)
        retVal = 0;
        for i = 1:m
            retVal = retVal + (h(X(i, :)) - y(i, 1)) * X(i, thetaIndex);
        end
        retVal = retVal * alpha / m;
    end

    for iter = 1:num_iters
        for thetaIndex = 1:size(theta, 1)
            newThetas(thetaIndex, 1) = newThetas(thetaIndex, 1) - gradientValue(thetaIndex);
        end

        theta = newThetas;
        J_history(iter, 1) = J(theta);
    end
end
