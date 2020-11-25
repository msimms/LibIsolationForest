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

function test_random(numTrees::UInt64, subsamplingsize::UInt64, numtrainingsamples::UInt64, numtests::UInt64)
    forest = IsolationForest::Forest(numTrees, subSamplingSize)

    # Note the time at which the test began.
    start_time = now()

    # Create some training samples.
    training_x = []
    training_y = []
    for i in range(0, num_training_samples)
        sample = IsolationForest::Sample()
        sample.name = "Training Sample " + i

        x = rand(Int, (0, 25))
        y = rand(Int, (0, 25))

        features = Dict([("x", x), ("y", y)])
        sample.add_features(features)
        forest.add_sample(sample)

        # So we can graph this later.
        training_x.append(x)
        training_y.append(y)
    end

    # Compute the elapsed time.
    elapsed_time = time.time() - start_time
end

function test_iris(numTrees::UInt64, subSamplingSize::UInt64)
    forest = IsolationForest::Forest(numTrees, subSamplingSize)

    avg_control_set_score = 0.0
    avg_outlier_set_score = 0.0
    avg_control_set_normalized_score = 0.0
    avg_outlier_set_normalized_score = 0.0
    num_control_tests = 0
    num_outlier_tests = 0

    # Note the time at which the test began.
    start_time = now()


    # Compute the elapsed time.
    elapsed_time = time.time() - start_time
end

function read_iris_data(fileName::String)
    data = []
    data = CSV.read(fileName)
    ts = data[1]
    x = data[2]
    y = data[3]
    z = data[4]
    ts, x, y, z
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

println("Test 1")
println("------")
avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time = test_random(10, 10, 100, 100)
println("Average of control test samples: ", avg_control_set_score)
println("Average of normalized control test samples: ", avg_control_set_normalized_score)
println("Average of outlier test samples: ", avg_outlier_set_score)
println("Average of normalized outlier test samples: ", avg_outlier_set_normalized_score)
println("Total time for Test 1: ", elapsed_time + " seconds.")
