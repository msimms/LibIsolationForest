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
import os
import random
import sys
import time
import IsolationForest
import plotly
import plotly.graph_objs as go

def test_random(num_trees, sub_sampling_size, num_training_samples, num_tests, plot):
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
    avg_normal_score = 0.0
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
        avg_normal_score = avg_normal_score + score
    avg_normal_score = avg_normal_score / num_tests

    # Test samples (different from training samples).
    outlier_x = []
    outlier_y = []
    avg_outlier_score = 0.0
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
        avg_outlier_score = avg_outlier_score + score
    avg_outlier_score = avg_outlier_score / num_tests

    # Compute the elapsed time.
    elapsed_time = time.time() - start_time

    # Create a trace.
    if plot:
        training_trace = go.Scatter(x=training_x, y=training_y, mode='markers', name='training')
        normal_trace = go.Scatter(x=normal_x, y=normal_y, mode='markers', name='normal')
        outlier_trace = go.Scatter(x=outlier_x, y=outlier_y, mode='markers', name='outlier')
        data = [training_trace, normal_trace, outlier_trace]
        plotly.offline.plot(data, filename='isolationforest_test.html')
    
    return avg_normal_score, avg_outlier_score, elapsed_time

def test_iris(num_trees, sub_sampling_size):
    forest = IsolationForest.Forest(num_trees, sub_sampling_size)

    avg_normal_score = 0.0
    avg_outlier_score = 0.0
    num_normal_tests = 0
    num_outlier_tests = 0

    # Note the time at which the test began.
    start_time = time.time()

    data_file_name = os.path.realpath(os.path.join(os.path.realpath(__file__), "..", "..", "data", "iris.data.txt"))
    if os.path.isfile(data_file_name):

        with open(data_file_name) as csv_file:
            training_class_name = 'Iris-setosa'
            test_samples = []

            # Each row in the file represents one sample. We'll use some for training and save some for test.
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) < 5:
                    continue

                features = []
                features.append({"sepal length cm": row[0]})
                features.append({"sepal width cm": row[1]})
                features.append({"petal length cm": row[2]})
                features.append({"petal width cm": row[3]})

                sample = IsolationForest.Sample(row[4])
                sample.add_features(features)

                if random.randint(0,10) > 0 and row[4] == training_class_name: # Use for training
                    forest.add_sample(sample)
                else: # Save for test
                    test_samples.append(sample)

            # Create the forest.
            forest.create()

            # Use each test sample.
            for test_sample in test_samples:
                score = forest.score(test_sample)
                if training_class_name == test_sample.name:
                    avg_normal_score = avg_normal_score + score
                    num_normal_tests = num_normal_tests + 1
                else:
                    avg_outlier_score = avg_outlier_score + score
                    num_outlier_tests = num_outlier_tests + 1

            if num_normal_tests > 0:
                avg_normal_score = avg_normal_score / num_normal_tests
            if num_outlier_tests > 0:
                avg_outlier_score = avg_outlier_score / num_outlier_tests

    else:
        print("Could not find " + data_file_name)

    # Compute the elapsed time.
    elapsed_time = time.time() - start_time

    return avg_normal_score, avg_outlier_score, elapsed_time

def main():
    # Parse command line options.
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", default=False, help="Plots the test data", required=False)

    try:
        args = parser.parse_args()
    except IOError as e:
        parser.error(e)
        sys.exit(1)

    print("Test 1")
    print("------")
    avg_normal_score, avg_outlier_score, elapsed_time = test_random(10, 10, 100, 100, args.plot)
    print("Average of normal test samples: %.4f" % avg_normal_score)
    print("Average of outlier test samples: %.4f" % avg_outlier_score)
    print("Total time for Test 1: %.4f" % elapsed_time + " seconds.\n")

    print("Test 2")
    print("------")
    avg_normal_score, avg_outlier_score, elapsed_time = test_random(100, 100, 1000, 100, args.plot)
    print("Average of normal test samples: %.4f" % avg_normal_score)
    print("Average of outlier test samples: %.4f" % avg_outlier_score)
    print("Total time for Test 2: %.4f" % elapsed_time + " seconds.\n")

    print("Test 3 (Iris Test)")
    print("------------------")
    avg_normal_score, avg_outlier_score, elapsed_time = test_iris(50, 50)
    print("Average of normal test samples: %.4f" % avg_normal_score)
    print("Average of outlier test samples: %.4f" % avg_outlier_score)
    print("Total time for Test 2: %.4f" % elapsed_time + " seconds.")

if __name__ == "__main__":
    main()
