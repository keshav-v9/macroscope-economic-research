function result = newey_west_ols(X, y, maxlags)
%NEWEY_WEST_OLS OLS with Bartlett-kernel HAC covariance.
%   Uses the statsmodels-compatible uncorrected sandwich estimator and
%   asymptotic normal inference, avoiding Econometrics Toolbox dependencies.

    arguments
        X double
        y double
        maxlags (1,1) double {mustBeInteger,mustBeNonnegative} = 4
    end

    [n, k] = size(X);
    assert(length(y) == n, 'X and y must have equal row counts.');
    beta = X \ y;
    residuals = y - X * beta;

    meat = zeros(k, k);
    for t = 1:n
        xt = X(t, :)';
        meat = meat + residuals(t)^2 * (xt * xt');
    end
    for lag = 1:maxlags
        weight = 1 - lag / (maxlags + 1);
        gamma = zeros(k, k);
        for t = (lag + 1):n
            xt = X(t, :)';
            xl = X(t - lag, :)';
            gamma = gamma + residuals(t) * residuals(t - lag) * (xt * xl');
        end
        meat = meat + weight * (gamma + gamma');
    end

    bread = inv(X' * X);
    covariance = bread * meat * bread;
    standardError = sqrt(diag(covariance));
    tStatistic = beta ./ standardError;
    pValue = erfc(abs(tStatistic) / sqrt(2));
    criticalValue = 1.959963984540054;

    result.beta = beta;
    result.covariance = covariance;
    result.standardError = standardError;
    result.tStatistic = tStatistic;
    result.pValue = pValue;
    result.ciLower = beta - criticalValue * standardError;
    result.ciUpper = beta + criticalValue * standardError;
    result.residuals = residuals;
end
