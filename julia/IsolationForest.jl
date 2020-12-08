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
    features::Dict
end

# Isolation Forest.
mutable struct Forest
    numTrees::UInt64
    subSamplingSize::UInt64
    featureValues::Dict # Dictionary that maps feature names to a sorted array of feature values
    trees::Array
end

# Adds the features to the specified sample.
function add_features_to_sample(sample::Sample, features::Dict)
    merge!(sample.features, features)
end

# Adds each of the sample's features to the list of known features with the corresponding set of unique values.
function add_sample_to_forest(forest::Forest, sample::Sample)

    # We don't store the sample directly, just the features.
    for feature_name in keys(sample.features)
        feature_value = sample.features[feature_name] # Value to insert

        try
            existing_feature_values = forest.featureValues[feature_name]
            push!(existing_feature_values, feature_value)
            sort(existing_feature_values)
        catch error
            if isa(error, KeyError)
                existing_feature_values = []
                push!(existing_feature_values, feature_value)
                forest.featureValues[feature_name] = existing_feature_values
            end
        end
    end
end

# Creates and returns a single tree. As this is a recursive function, depth indicates the current depth of the recursion.
function create_tree(forest::Forest, feature_values::Array, depth::UInt64)

    # Sanity check
    feature_values_len = len(feature_values)
    if feature_values_len <= 1
        return Nothing
    end

    # If we've exceeded the maximum desired depth, then stop.
    if (forest.subSamplingSize > 0) && (depth >= forest.subSamplingSize)
        return Nothing
    end

    # Randomly select a feature.
    selected_feature_index = rand(1:feature_values_len)
    selected_feature_name = keys(forest.feature_values)[selected_feature_index]

    # Randomly select a split value.
    feature_value_set = forest.feature_values[selected_feature_name]
    feature_value_set_len = length(feature_value_set)
    if feature_value_set_len <= 1
        return Nothing
    end
    split_value_index = rand(0:feature_value_set_len - 1)
    split_value = feature_value_set[split_value_index]

    # Create a tree node to hold the split value.
    tree = Node(selected_feature_name, split_value, nothing, nothing)

    # Create two versions of the feature value set that we just used,
    # one for the left side of the tree and one for the right.
    temp_feature_values = forest.feature_values
end

# Scores the sample against the specified tree.
function score_sample_against_tree(tree::Node, sample::Sample)
    depth = 0.0
    current_node = tree

    while current_node != missing
        found_feature = false

        # Find the next feature in the sample.
        for current_feature in sample.features
            current_feature_name = list(current_feature)[0]

            # If the current node has the feature in question.
            if current_feature_name == current_node.feature_name
                current_feature_value = list(current_feature.values())[0]
                if current_feature_value < current_node.split_value
                    current_node = current_node.left
                else
                    current_node = current_node.right
                end

                depth = depth + 1.0
                found_feature = True
                break
            end
        end

        # If the tree contained a feature not in the sample then take
        # both sides of the tree and average the scores together.
        if not found_feature
            left_depth = depth + score_sample_against_tree(sample, current_node.left)
            right_depth = depth + score_sample_against_tree(sample, current_node.right)
            return (left_depth + right_depth) / 2.0
        end
    end

    return depth
end

# Scores the sample against the entire forest of trees. Result is the average path length.
function score_sample_against_forest(forest::Forest, sample::Sample)
    num_trees = 0
    avg_path_len = 0.0

    for tree in forest.trees
        path_len = score_sample_against_tree(tree, sample)
        if path_len > 0
            avg_path_len = avg_path_len + path_len
            num_trees = num_trees + 1
        end
    end

    if num_trees > 0
        avg_path_len = avg_path_len / num_trees
    end

    return avg_path_len
end

# Scores the sample against the entire forest of trees. Result is normalized so that values
# close to 1 indicate anomalies and values close to zero indicate normal values.
function forest_normalized_score(forest::Forest, sample::Sample)

    # Compute the average path length for all valid trees.
    num_trees = 0
    avg_path_len = 0.0

    for tree in forest.trees
        path_len = score_sample_against_tree(tree, sample)
        if path_len > 0
            avg_path_len = avg_path_len + path_len
            num_trees = num_trees + 1
        end
    end

    if num_trees > 0
        avg_path_len = avg_path_len / num_trees
    end

    # Normalize, per the original paper.
    score = 0.0
    if num_trees > 1.0
    end

    return score
end

end
