#	MIT License
#
#  Copyright © 2020 Michael J Simms. All rights reserved.
#
#	Permission is hereby granted, free of charge, to any person obtaining a copy
#	of this software and associated documentation files (the "Software"), to deal
#	in the Software without restriction, including without limitation the rights
#	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#	copies of the Software, and to permit persons to whom the Software is
#	furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in all
#	copies or substantial portions of the Software.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#	SOFTWARE.

include("IsolationForest.jl")

using Pkg
Pkg.add("ArgParse")
using ArgParse
Pkg.add("CSV")
using CSV
Pkg.add("Dates")
using Dates

function test_random(num_trees::Int64, sub_sampling_size::Int64, num_training_samples::Int64, num_tests::Int64)
    forest = IsolationForest.Forest(num_trees, sub_sampling_size, Dict(), [])

    # Note the time at which the test began.
    start_time = now()

    # Create some training samples.
    training_x = []
    training_y = []
    for i = 0:num_training_samples
        sample_name = string("Training Sample ", i)
        sample = IsolationForest.Sample(sample_name, Dict())

        x = rand(0:25)
        y = rand(0:25)

        features = Dict("x" => x, "y" => y)
        IsolationForest.add_features_to_sample(sample, features)
        IsolationForest.add_sample_to_forest(forest, sample)

        # So we can graph this later.
        push!(training_x, x)
        push!(training_y, y)
    end

    # Test samples (similar to training samples).
    normal_x = []
    normal_y = []
    avg_control_set_score = 0.0
    avg_control_set_normalized_score = 0.0
    for i = 0:num_tests
        sample_name = string("Normal Sample ", i)
        sample = IsolationForest.Sample(sample_name, Dict())

        x = rand(0:25)
        y = rand(0:25)

        features = Dict("x" => x, "y" => y)
        IsolationForest.add_features_to_sample(sample, features)

        # So we can graph this later.
        push!(normal_x, x)
        push!(normal_y, y)

        # Run a test with the sample that doesn't contain outliers.
        score = IsolationForest.score_sample_against_forest(forest, sample)
        normalized_score = IsolationForest.forest_normalized_score(forest, sample)
        avg_control_set_score = avg_control_set_score + score
        avg_control_set_normalized_score = avg_control_set_normalized_score + normalized_score
    end
    avg_control_set_score = avg_control_set_score / num_tests
    avg_control_set_normalized_score = avg_control_set_normalized_score / num_tests

    # Test samples (different from training samples).
    outlier_x = []
    outlier_y = []
    avg_outlier_set_score = 0.0
    avg_outlier_set_normalized_score = 0.0
    for i = 0:num_tests
        sample_name = string("Outlier Sample ", i)
        sample = IsolationForest.Sample(sample_name, Dict())

        x = rand(20:45)
        y = rand(20:45)

        features = Dict("x" => x, "y" => y)
        IsolationForest.add_features_to_sample(sample, features)

        # So we can graph this later.
        push!(outlier_x, x)
        push!(outlier_y, y)

        # Run a test with the sample that doesn't contain outliers.
        score = IsolationForest.score_sample_against_forest(forest, sample)
        normalized_score = IsolationForest.forest_normalized_score(forest, sample)
        avg_outlier_set_score = avg_outlier_set_score + score
        avg_outlier_set_normalized_score = avg_outlier_set_normalized_score + normalized_score
    end
    avg_outlier_set_score = avg_outlier_set_score / num_tests
    avg_outlier_set_normalized_score = avg_outlier_set_normalized_score / num_tests

    # Compute the elapsed time.
    elapsed_time = now() - start_time

    return avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time
end

function test_iris(num_trees::Int64, sub_sampling_size::Int64)
    forest = IsolationForest.Forest(num_trees, sub_sampling_size, Dict(), [])

    avg_control_set_score = 0.0
    avg_outlier_set_score = 0.0
    avg_control_set_normalized_score = 0.0
    avg_outlier_set_normalized_score = 0.0
    num_control_tests = 0
    num_outlier_tests = 0

    # Note the time at which the test began.
    start_time = now()

    data_file_name = realpath(joinpath(dirname(@__FILE__), "..", "data", "iris.data.txt"))
    data = CSV.read(data_file_name)

    training_class_name = "Iris-setosa"
    test_samples = []

    sl = data[1]
    sw = data[2]
    pl = data[3]
    pw = data[4]
    names = data[5]

    # Each row in the file represents one sample. We'll use some for training and save some for test.
    for i = 1:length(names)
        features = Dict("sepal length cm" => sl[i], "sepal width cm" => sw[i], "petal length cm" => pl[i], "petal width cm" => pw[i], "name" => names[i])
        sample = IsolationForest.Sample("Iris Data", features)
    end

    # Compute the elapsed time.
    elapsed_time = now() - start_time

    return avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time
end

# Parses the command line arguments
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--csv"
            help = "another option with an argument"
            arg_type = String
            default = "data/10_pullups.csv"
    end

    return parse_args(s)
end

# Parse command line options.
parsed_args = parse_commandline()

println("Test 1")
println("------")
avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time = test_random(10, 10, 100, 100)
println("Average of control test samples: ", avg_control_set_score)
println("Average of normalized control test samples: ", avg_control_set_normalized_score)
println("Average of outlier test samples: ", avg_outlier_set_score)
println("Average of normalized outlier test samples: ", avg_outlier_set_normalized_score)
println("Total time for Test 1: ", elapsed_time, ".\n")

println("Test 2")
println("------")
avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time = test_random(100, 100, 1000, 100)
println("Average of control test samples: ", avg_control_set_score)
println("Average of normalized control test samples: ", avg_control_set_normalized_score)
println("Average of outlier test samples: ", avg_outlier_set_score)
println("Average of normalized outlier test samples: ", avg_outlier_set_normalized_score)
println("Total time for Test 2: ", elapsed_time, ".\n")

println("Test 3 (Iris Test)")
println("------------------")
avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time = test_iris(50, 50)
println("Average of control test samples: ", avg_control_set_score)
println("Average of normalized control test samples: ", avg_control_set_normalized_score)
println("Average of outlier test samples: ", avg_outlier_set_score)
println("Average of normalized outlier test samples: ", avg_outlier_set_normalized_score)
println("Total time for Test 3 (Iris Data): ", elapsed_time, ".\n")
