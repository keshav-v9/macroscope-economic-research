function tests = test_macroscope
%TEST_MACROSCOPE MATLAB integration and metric regression tests.
    tests = functiontests(localfunctions);
end

function testMetricDefinition(testCase)
    result = forecast_metrics([1;-1;3],[0;0;1]);
    verifyEqual(testCase,result.mae,4/3,AbsTol=1e-12);
    verifyEqual(testCase,result.rmse,sqrt(2),AbsTol=1e-12);
end

function testCompletePipelineAndTemporalAudit(testCase)
    outputDir = string(tempname);
    mkdir(outputDir);
    cleanup = onCleanup(@() rmdir(outputDir,"s")); %#ok<NASGU>
    metrics = run_analysis("data/macrodata.csv",outputDir,80,80);

    verifyEqual(testCase,metrics.source_observations,203);
    verifyEqual(testCase,metrics.complete_model_observations,198);
    verifyEqual(testCase,metrics.evaluation_observations,115);
    verifyEqual(testCase,metrics.model.rmse,0.9238082433771755,AbsTol=1e-9);
    verifyEqual(testCase,metrics.no_change_baseline.rmse,1.0514999638198013,AbsTol=1e-9);

    predictions = readtable(fullfile(outputDir,"forecast_predictions.csv"),TextType="string");
    verifyEqual(testCase,predictions.baseline_prediction,zeros(115,1));
    origin = quarter_ordinal_for_test(predictions.origin_quarter);
    latest = quarter_ordinal_for_test(predictions.latest_known_target);
    verifyTrue(testCase,all(latest <= origin));
end

function ordinal = quarter_ordinal_for_test(labels)
    ordinal = 4 .* str2double(extractBefore(labels,"Q")) + ...
        str2double(extractAfter(labels,"Q"));
end
