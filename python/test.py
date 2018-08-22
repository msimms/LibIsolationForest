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
import IsolationForest
import plotly.plotly as py
import plotly.graph_objs as go

def main():
    num_trees = 10
    sub_sampling_size = 10
    forest = IsolationForest.Forest(num_trees, sub_sampling_size)
    num_training_samples = 100
    num_tests = 10

    # Create some training samples.
    training_x = []
    training_y = []
    for i in range(0,num_training_samples):
        sample = IsolationForest.Sample("training")
        features = []

        x = 0.3 * (random.randint(0,100))
        y = 0.3 * (random.randint(0,100))

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
    print "Test samples that are similar to the training set."
    print "--------------------------------------------------"
    normal_x = []
    normal_y = []
    avg_normal_score = 0.0
    for i in range(0,num_tests):
        sample = IsolationForest.Sample("normal sample")
        features = []

        x = 0.3 * (random.randint(0,100))
        y = 0.3 * (random.randint(0,100))

        features.append({"x": x})
        features.append({"y": y})
        sample.add_features(features)

        # So we can graph this later.
        normal_x.append(x)
        normal_y.append(y)

        # Run a test with the sample that doesn't contain outliers.
        score = forest.score(sample)
        avg_normal_score = avg_normal_score + score
        print "Normal test sample " + str(i) + ": " + str(score)
    avg_normal_score = avg_normal_score / num_tests

    # Test samples (similar to training samples).
    print "Test samples that are different to the training set."
    print "----------------------------------------------------"
    outlier_x = []
    outlier_y = []
    avg_outlier_score = 0.0
    for i in range(0,num_tests):
        sample = IsolationForest.Sample("outlier sample")
        features = []

        x = 0.3 * (random.randint(0,100))
        y = 0.3 * (random.randint(0,100))
        sample.add_features(features)

        features.append({"x": x})
        features.append({"y": y})

        # So we can graph this later.
        outlier_x.append(x)
        outlier_y.append(y)

        # Run a test with the sample that doesn't contain outliers.
        score = forest.score(sample)
        avg_outlier_score = avg_outlier_score + score
        print "Outlier test sample " + str(i) + ": " + str(score)
    avg_outlier_score = avg_outlier_score / num_tests

    # Create a trace.
    training_trace = go.Scatter(x = training_x, y = training_y, mode = 'markers')
    normal_trace = go.Scatter(x = normal_x, y = normal_y, mode = 'markers')
    outlier_trace = go.Scatter(x = outlier_x, y = outlier_y, mode = 'markers')
    data = [training_trace, normal_trace, outlier_trace]
    py.iplot(data, filename='basic-scatter')

if __name__ == "__main__":
    main()
