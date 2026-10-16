function metrics = forecast_metrics(actual, predicted)
%FORECAST_METRICS Mean absolute error and root mean squared error.
    arguments
        actual double
        predicted double
    end
    assert(~isempty(actual) && isequal(size(actual), size(predicted)), ...
        'Inputs must be nonempty and have matching dimensions.');
    errors = actual - predicted;
    metrics.mae = mean(abs(errors));
    metrics.rmse = sqrt(mean(errors .^ 2));
end
