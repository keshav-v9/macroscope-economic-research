function metrics = run_analysis(inputCsv, outputDir, minTrainSize, rollingWindow)
%RUN_ANALYSIS Execute MacroScope econometrics, forecasts, and visualization.
%
% Example from the repository root:
%   addpath('matlab'); run_analysis('data/macrodata.csv','outputs',80,80)

    arguments
        inputCsv (1,1) string = "data/macrodata.csv"
        outputDir (1,1) string = "outputs"
        minTrainSize (1,1) double {mustBeInteger,mustBePositive} = 80
        rollingWindow (1,1) double {mustBeInteger,mustBePositive} = 80
    end

    if ~isfolder(outputDir), mkdir(outputDir); end
    raw = readtable(inputCsv, TextType="string");
    required = ["quarter_label","quarter_index","realgdp","infl","tbilrate","unemp"];
    assert(all(ismember(required, string(raw.Properties.VariableNames))), ...
        'Input CSV is missing required fields.');
    assert(height(raw) == 203, 'Expected 203 source observations.');
    assert(all(diff(raw.quarter_index) == 1), 'Quarters must be sorted and contiguous.');
    assert(numel(unique(raw.quarter_label)) == height(raw), 'Quarters must be unique.');
    assert(~any(any(ismissing(raw(:, required)))), ...
        'Required source fields contain missing data.');

    gdpGrowth = [NaN; 100 .* diff(log(raw.realgdp))];
    target = [raw.unemp(5:end) - raw.unemp(1:end-4); NaN(4,1)];
    targetIndex = raw.quarter_index + 4;
    targetQuarter = strings(height(raw), 1);
    targetQuarter(1:end-4) = raw.quarter_label(5:end);

    valid = ~isnan(gdpGrowth) & ~isnan(target);
    modelData = table(raw.quarter_label(valid), raw.quarter_index(valid), ...
        targetQuarter(valid), targetIndex(valid), raw.unemp(valid), raw.infl(valid), ...
        raw.tbilrate(valid), gdpGrowth(valid), target(valid), ...
        VariableNames=["origin_quarter","origin_index","target_quarter", ...
        "target_index","unemp","infl","tbilrate","gdp_growth", ...
        "unemp_change_4q"]);
    assert(height(modelData) == 198, 'Expected 198 complete modeling rows.');

    predictorNames = ["unemp","infl","tbilrate","gdp_growth"];
    X = [ones(height(modelData),1), modelData{:, predictorNames}];
    y = modelData.unemp_change_4q;
    full = newey_west_ols(X, y, 4);
    terms = ["const"; predictorNames'];
    regressionTable = table(terms, full.beta, full.standardError, ...
        full.tStatistic, full.pValue, full.ciLower, full.ciUpper, ...
        VariableNames=["term","coefficient","hac_std_error","t_statistic", ...
        "p_value","ci_95_lower","ci_95_upper"]);
    writetable(regressionTable, fullfile(outputDir, "regression_table.csv"));

    records = table();
    for i = 1:height(modelData)
        originIndex = modelData.origin_index(i);
        eligible = modelData.target_index <= originIndex;
        train = modelData(eligible, :);
        if height(train) < minTrainSize, continue; end

        trainX = [ones(height(train),1), train{:, predictorNames}];
        beta = trainX \ train.unemp_change_4q;
        originX = [1, modelData{i, predictorNames}];
        prediction = originX * beta;

        rollingTrain = train(max(1, height(train)-rollingWindow+1):end, :);
        rollingX = [ones(height(rollingTrain),1), rollingTrain{:, predictorNames}];
        rollingBeta = rollingX \ rollingTrain.unemp_change_4q;
        rollingPrediction = originX * rollingBeta;
        historicalMean = mean(train.unemp_change_4q);
        truth = modelData.unemp_change_4q(i);

        next = table(modelData.origin_quarter(i), modelData.target_quarter(i), ...
            train.origin_quarter(1), train.origin_quarter(end), train.target_quarter(end), ...
            height(train), truth, prediction, 0.0, historicalMean, rollingPrediction, ...
            truth-prediction, truth, truth-historicalMean, truth-rollingPrediction, ...
            VariableNames=["origin_quarter","target_quarter","train_start_origin", ...
            "train_end_origin","latest_known_target","train_size","y_true", ...
            "model_prediction","baseline_prediction","historical_mean_prediction", ...
            "rolling_prediction","model_error","baseline_error", ...
            "historical_mean_error","rolling_error"]);
        if isempty(records)
            records = next;
        else
            records = [records; next]; %#ok<AGROW>
        end
    end
    assert(~isempty(records), 'No origins meet the minimum training size.');
    assert(all(quarter_ordinal(records.latest_known_target) <= ...
        quarter_ordinal(records.origin_quarter)), ...
        'Temporal leakage detected in a training window.');
    writetable(records, fullfile(outputDir, "forecast_predictions.csv"));

    modelMetrics = forecast_metrics(records.y_true, records.model_prediction);
    baselineMetrics = forecast_metrics(records.y_true, records.baseline_prediction);
    historicalMetrics = forecast_metrics(records.y_true, records.historical_mean_prediction);
    rollingMetrics = forecast_metrics(records.y_true, records.rolling_prediction);
    improvement.mae = 100 * (baselineMetrics.mae-modelMetrics.mae) / baselineMetrics.mae;
    improvement.rmse = 100 * (baselineMetrics.rmse-modelMetrics.rmse) / baselineMetrics.rmse;

    metrics.evaluation_observations = height(records);
    metrics.first_origin = records.origin_quarter(1);
    metrics.last_origin = records.origin_quarter(end);
    metrics.first_target = records.target_quarter(1);
    metrics.last_target = records.target_quarter(end);
    metrics.model = modelMetrics;
    metrics.no_change_baseline = baselineMetrics;
    metrics.improvement_percent = improvement;
    metrics.source_observations = height(raw);
    metrics.complete_model_observations = height(modelData);
    metrics.forecast_horizon_quarters = 4;
    metrics.minimum_training_observations = minTrainSize;
    metrics.rolling_window_quarters = rollingWindow;
    write_text(fullfile(outputDir,"metrics.json"), pretty_json(metrics));

    model = repmat(["expanding_ols";"no_change";"historical_mean";"rolling_80q_ols"],1,1);
    mae = [modelMetrics.mae;baselineMetrics.mae;historicalMetrics.mae;rollingMetrics.mae];
    rmse = [modelMetrics.rmse;baselineMetrics.rmse;historicalMetrics.rmse;rollingMetrics.rmse];
    n_forecasts = repmat(height(records),4,1);
    writetable(table(model,mae,rmse,n_forecasts), fullfile(outputDir,"robustness_metrics.csv"));

    create_plots(raw, gdpGrowth, records, regressionTable, outputDir);
    fprintf('MATLAB analysis complete: %d forecasts, RMSE %.4f vs %.4f baseline.\n', ...
        height(records), modelMetrics.rmse, baselineMetrics.rmse);
end

function create_plots(raw, gdpGrowth, records, coefficients, outputDir)
    dates = datetime(raw.year, 3*raw.quarter, 1);
    fig = figure(Visible="off", Position=[100 100 1200 720]);
    layout = tiledlayout(2,2,TileSpacing="compact");
    series = {raw.unemp,raw.infl,raw.tbilrate,gdpGrowth};
    titles = {"Unemployment rate","Inflation rate","3-month Treasury bill rate", ...
        "Real GDP quarter-over-quarter log growth"};
    units = {"Percent","Annualized percent","Percent","Percent"};
    for i=1:4
        nexttile; plot(dates,series{i},LineWidth=1.2); yline(0,Color=[.5 .5 .5]); grid on;
        title(titles{i}); ylabel(units{i});
    end
    title(layout,"U.S. macroeconomic indicators, 1959Q1-2009Q3");
    exportgraphics(fig,fullfile(outputDir,"macro_indicators.png"),Resolution=160); close(fig);

    targetDates = quarter_dates(records.target_quarter);
    fig = figure(Visible="off", Position=[100 100 1200 550]);
    plot(targetDates,records.y_true,LineWidth=1.8); hold on;
    plot(targetDates,records.model_prediction,LineWidth=1.3);
    plot(targetDates,records.baseline_prediction,"--",LineWidth=1.2); grid on;
    title("Four-quarter-ahead unemployment-change forecasts");
    xlabel("Target quarter"); ylabel("Percentage points");
    legend("Actual 4-quarter change","Expanding-window OLS","No-change baseline",Location="best");
    exportgraphics(fig,fullfile(outputDir,"forecast_comparison.png"),Resolution=160); close(fig);

    fig = figure(Visible="off", Position=[100 100 900 500]);
    errorbar(coefficients.coefficient,1:height(coefficients), ...
        coefficients.coefficient-coefficients.ci_95_lower, ...
        coefficients.ci_95_upper-coefficients.coefficient,"o","horizontal");
    xline(0,"--"); grid on; yticks(1:height(coefficients)); yticklabels(coefficients.term);
    title("OLS coefficients with 95% HAC confidence intervals");
    xlabel("Effect on 4-quarter unemployment change (percentage points)");
    ylabel("Predictor at forecast origin");
    exportgraphics(fig,fullfile(outputDir,"regression_coefficients.png"),Resolution=160); close(fig);
end

function dates = quarter_dates(labels)
    years = str2double(extractBefore(labels,"Q"));
    quarters = str2double(extractAfter(labels,"Q"));
    dates = datetime(years,3*quarters,1);
end

function ordinal = quarter_ordinal(labels)
    years = str2double(extractBefore(labels,"Q"));
    quarters = str2double(extractAfter(labels,"Q"));
    ordinal = 4 .* years + quarters;
end

function value = pretty_json(data)
    value = jsonencode(data,PrettyPrint=true) + newline;
end

function write_text(path, value)
    file = fopen(path,"w");
    assert(file ~= -1,"Could not open output file: %s",path);
    cleanup = onCleanup(@() fclose(file)); %#ok<NASGU>
    fprintf(file,"%s",value);
end
