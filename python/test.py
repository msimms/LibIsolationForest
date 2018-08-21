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

forest = IsolationForest.Forest(10, 10)

# Create some training samples.
for i in range(0,100):
    sample = IsolationForest.Sample("")
    features = []

    x = 0.3 * (random.randint(0,100))
    y = 0.3 * (random.randint(0,100))

    features.append({"x": x})
    features.append({"y": y})

    sample.add_features(features)
    forest.add_sample(sample)

# Create the isolation forest.
forest.create()

# Test samples (similar to training samples).
print "Test samples that are similar to the training set."
print "--------------------------------------------------"
for i in range(0,10):
    sample = IsolationForest.Sample("")
    features = []

    x = 0.3 * (random.randint(0,100))
    y = 0.3 * (random.randint(0,100))

    features.append({"x": x})
    features.append({"y": y})

    sample.add_features(features)

    # Run a test with the sample that doesn't contain outliers.
    score = forest.score(sample)
    print "Normal test sample " + str(i) + ": " + str(score)

print "\n"
