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

import random
import time
import IsolationForest
import plotly
import plotly.graph_objs as go

def test1():
    num_trees = 10
    sub_sampling_size = 10
    forest = IsolationForest.Forest(num_trees, sub_sampling_size)
    num_training_samples = 100
    num_tests = 10

    # Note the time at which the test began.
    start_time = time.time()

    # Create some training samples.
    training_x = []
    training_y = []
    for i in range(0,num_training_samples):
        sample = IsolationForest.Sample("training")
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
        sample = IsolationForest.Sample("normal sample")
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
        sample = IsolationForest.Sample("outlier sample")
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
    training_trace = go.Scatter(x=training_x, y=training_y, mode='markers', name='training')
    normal_trace = go.Scatter(x=normal_x, y=normal_y, mode='markers', name='normal')
    outlier_trace = go.Scatter(x=outlier_x, y=outlier_y, mode='markers', name='outlier')
    data = [training_trace, normal_trace, outlier_trace]
    plotly.offline.plot(data, filename='isolationforest_test.html')

    return avg_normal_score, avg_outlier_score, elapsed_time

def main():
    print("Test 1")
    print("------")
    avg_normal_score, avg_outlier_score, elapsed_time = test1()
    print("Average of normal test samples: " + str(avg_normal_score))
    print("Average of outlier test samples: " + str(avg_outlier_score))
    print("Total time for Test 1: " + str(elapsed_time) + " seconds.")

if __name__ == "__main__":
    main()
