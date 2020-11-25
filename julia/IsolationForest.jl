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

module IsolationForest

# Tree node, used internally.
mutable struct Node
    featureName::String
    splitValue::UInt64
    left::Node
    right::Node
end

# This struct represents a sample. Each sample has a name and list of features.
mutable struct Sample
    name::String
    features::Array
end

# Isolation Forest.
mutable struct Forest
    numTrees::UInt64
    subSamplingSize::UInt64
    featureValues::Array
    trees::Array
end

function initializeForest(forest::Forest)
end

# Adds each of the sample's features to the list of known features with the corresponding set of unique values.
function addSample(forest::Forest, sample::Sample)

    # We don't store the sample directly, just the features.
    for feature in sample.features
    end
end

end
