#! /usr/bin/env python
#  MIT License
#
#  Copyright (c) 2018 Michael J Simms. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import argparse
import csv
import json
import os
import random
import sys
import time
from isolationforest import IsolationForest

def test_random(num_trees, sub_sampling_size, num_training_samples, num_tests, plot, plot_filename):
    forest = IsolationForest.Forest(num_trees, sub_sampling_size)

    # Note the time at which the test began.
    start_time = time.time()

    # Create some training samples.
    training_x = []
    training_y = []
    for i in range(0,num_training_samples):
        sample = IsolationForest.Sample("Training Sample " + str(i))
        features = []

        x = random.randint(0,25)
        y = random.randint(0,25)

        features.append({"x": x})
        features.append({"y": y})
        sample.add_features(features)
        forest.add_sample(sample)

        # So we can graph this later.
        training_x.append(x)
        training_y.append(y)

    # Create the isolation forest.
    forest.create()

    # Test samples (similar to training samples).
    normal_x = []
    normal_y = []
    avg_control_set_score = 0.0
    avg_control_set_normalized_score = 0.0
    for i in range(0,num_tests):
        sample = IsolationForest.Sample("Normal Sample " + str(i))
        features = []

        x = random.randint(0,25)
        y = random.randint(0,25)

        features.append({"x": x})
        features.append({"y": y})
        sample.add_features(features)

        # So we can graph this later.
        normal_x.append(x)
        normal_y.append(y)

        # Run a test with the sample that doesn't contain outliers.
        score = forest.score(sample)
        normalized_score = forest.normalized_score(sample)
        avg_control_set_score = avg_control_set_score + score
        avg_control_set_normalized_score = avg_control_set_normalized_score + normalized_score
    avg_control_set_score = avg_control_set_score / num_tests
    avg_control_set_normalized_score = avg_control_set_normalized_score / num_tests

    # Test samples (different from training samples).
    outlier_x = []
    outlier_y = []
    avg_outlier_set_score = 0.0
    avg_outlier_set_normalized_score = 0.0
    for i in range(0,num_tests):
        sample = IsolationForest.Sample("Outlier Sample " + str(i))
        features = []

        x = random.randint(20,45)
        y = random.randint(20,45)

        features.append({"x": x})
        features.append({"y": y})
        sample.add_features(features)

        # So we can graph this later.
        outlier_x.append(x)
        outlier_y.append(y)

        # Run a test with the sample that doesn't contain outliers.
        score = forest.score(sample)
        normalized_score = forest.normalized_score(sample)
        avg_outlier_set_score = avg_outlier_set_score + score
        avg_outlier_set_normalized_score = avg_outlier_set_normalized_score + normalized_score
    avg_outlier_set_score = avg_outlier_set_score / num_tests
    avg_outlier_set_normalized_score = avg_outlier_set_normalized_score / num_tests

    # Compute the elapsed time.
    elapsed_time = time.time() - start_time

    # Create a trace.
    if plot:
        import plotly
        import plotly.graph_objs as go

        training_trace = go.Scatter(x=training_x, y=training_y, mode='markers', name='training')
        normal_trace = go.Scatter(x=normal_x, y=normal_y, mode='markers', name='normal')
        outlier_trace = go.Scatter(x=outlier_x, y=outlier_y, mode='markers', name='outlier')
        data = [training_trace, normal_trace, outlier_trace]
        plotly.offline.plot(data, filename=plot_filename)
    
    return avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time

def test_iris(num_trees, sub_sampling_size, plot, dump, load):
    FEATURE_SEPAL_LENGTH_CM = "sepal length cm"
    FEATURE_SEPAL_WIDTH_CM = "sepal width cm"
    FEATURE_PETAL_LENGTH_CM = "petal length cm"
    FEATURE_PETAL_WIDTH_CM = "petal width cm"

    forest = IsolationForest.Forest(num_trees, sub_sampling_size)

    avg_control_set_score = 0.0
    avg_outlier_set_score = 0.0
    avg_control_set_normalized_score = 0.0
    avg_outlier_set_normalized_score = 0.0
    num_control_tests = 0
    num_outlier_tests = 0

    # Test loading a forest from file.
    if load:
        with open('isolationforest_test_iris.json', 'rt') as json_file:
            json_str = json_file.read()
            json_data = json.loads(json_str)
            forest.load(json_data)

    # Note the time at which the test began.
    start_time = time.time()

    data_file_name = os.path.realpath(os.path.join(os.path.realpath(__file__), "..", "..", "data", "iris.data.txt"))
    if os.path.isfile(data_file_name):

        with open(data_file_name) as csv_file:
            training_class_name = 'Iris-setosa'
            training_samples = []
            test_samples = []

            # Each row in the file represents one sample. We'll use some for training and save some for test.
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:

                # Check for junk.
                if len(row) < 5:
                    continue

                features = []
                features.append({FEATURE_SEPAL_LENGTH_CM: float(row[0])})
                features.append({FEATURE_SEPAL_WIDTH_CM: float(row[1])})
                features.append({FEATURE_PETAL_LENGTH_CM: float(row[2])})
                features.append({FEATURE_PETAL_WIDTH_CM: float(row[3])})

                sample = IsolationForest.Sample(row[4])
                sample.add_features(features)

                # Randomly split the samples into training and test samples.
                if random.randint(0,10) > 5 and row[4] == training_class_name: # Use for training
                    if not load: # We loaded the forest from a file, so don't modify it here.
                        forest.add_sample(sample)
                    training_samples.append(sample)
                else: # Save for test
                    test_samples.append(sample)

            # Create the forest.
            forest.create()

            # Use each test sample.
            for test_sample in test_samples:
                score = forest.score(test_sample)
                normalized_score = forest.normalized_score(test_sample)
                if training_class_name == test_sample.name:
                    avg_control_set_score = avg_control_set_score + score
                    avg_control_set_normalized_score = avg_control_set_normalized_score + normalized_score
                    num_control_tests = num_control_tests + 1
                else:
                    avg_outlier_set_score = avg_outlier_set_score + score
                    avg_outlier_set_normalized_score = avg_outlier_set_normalized_score + normalized_score
                    num_outlier_tests = num_outlier_tests + 1

            # Compute statistics.
            if num_control_tests > 0:
                avg_control_set_score = avg_control_set_score / num_control_tests
                avg_control_set_normalized_score = avg_control_set_normalized_score / num_control_tests
            if num_outlier_tests > 0:
                avg_outlier_set_score = avg_outlier_set_score / num_outlier_tests
                avg_outlier_set_normalized_score = avg_outlier_set_normalized_score / num_outlier_tests

            # Compute the elapsed time.
            elapsed_time = time.time() - start_time

            # Create a trace.
            if plot:
                import plotly
                import plotly.graph_objs as go

                training_x = []
                training_y = []
                test_x = []
                test_y = []

                for sample in training_samples:
                    training_x.append(sample.features[FEATURE_SEPAL_LENGTH_CM])
                    training_y.append(sample.features[FEATURE_SEPAL_WIDTH_CM])
                for sample in test_samples:
                    test_x.append(sample.features[FEATURE_SEPAL_LENGTH_CM])
                    test_y.append(sample.features[FEATURE_SEPAL_WIDTH_CM])

                training_trace = go.Scatter(x=training_x, y=training_y, mode='markers', name='training')
                test_trace = go.Scatter(x=test_x, y=test_y, mode='markers', name='test')
                data = [training_trace, test_trace]
                plotly.offline.plot(data, filename='isolationforest_test_iris.html')

            # Dump the training data.
            if dump:
                json_data = forest.dump()
                with open('isolationforest_test_iris.json', 'wt') as json_file:
                    json_file.write(json.dumps(json_data))

    else:
        print("Could not find " + data_file_name)

    return avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time

def main():
    # Parse command line options.
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", default=False, help="Plots the test data", required=False)
    parser.add_argument("--dump", action="store_true", default=False, help="Dumps the forest data to a file", required=False)
    parser.add_argument("--load", action="store_true", default=False, help="Loads the forest data from a file", required=False)

    try:
        args = parser.parse_args()
    except IOError as e:
        parser.error(e)
        sys.exit(1)

    print("Test 1")
    print("------")
    avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time = test_random(10, 10, 100, 100, args.plot, 'isolationforest_test_1.html')
    print("Average of control test samples: %.4f" % avg_control_set_score)
    print("Average of normalized control test samples: %.4f" % avg_control_set_normalized_score)
    print("Average of outlier test samples: %.4f" % avg_outlier_set_score)
    print("Average of normalized outlier test samples: %.4f" % avg_outlier_set_normalized_score)
    print("Total time for Test 1: %.4f" % elapsed_time + " seconds.\n")

    print("Test 2")
    print("------")
    avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time = test_random(100, 100, 1000, 100, args.plot, 'isolationforest_test_2.html')
    print("Average of control test samples: %.4f" % avg_control_set_score)
    print("Average of normalized control test samples: %.4f" % avg_control_set_normalized_score)
    print("Average of outlier test samples: %.4f" % avg_outlier_set_score)
    print("Average of normalized outlier test samples: %.4f" % avg_outlier_set_normalized_score)
    print("Total time for Test 2: %.4f" % elapsed_time + " seconds.\n")

    print("Test 3 (Iris Test)")
    print("------------------")
    avg_control_set_score, avg_control_set_normalized_score, avg_outlier_set_score, avg_outlier_set_normalized_score, elapsed_time = test_iris(50, 50, args.plot, args.dump, args.load)
    print("Average of control test samples: %.4f" % avg_control_set_score)
    print("Average of normalized control test samples: %.4f" % avg_control_set_normalized_score)
    print("Average of outlier test samples: %.4f" % avg_outlier_set_score)
    print("Average of normalized outlier test samples: %.4f" % avg_outlier_set_normalized_score)
    print("Total time for Test 3 (Iris Test): %.4f" % elapsed_time + " seconds.")

if __name__ == "__main__":
    main()
