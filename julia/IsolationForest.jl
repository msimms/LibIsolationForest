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
    splitValue::Float64
    left::Any
    right::Any
end

# This struct represents a sample. Each sample has a name and list of features.
mutable struct Sample
    name::String
    features::Dict{String,Float64}
end

# Isolation Forest.
mutable struct Forest
    numTrees::UInt64
    subSamplingSize::UInt64
    featureValues::Dict{String,Array} # Dictionary that maps feature names to a sorted array of feature values
    trees::Array
end

# Adds the features to the specified sample.
function add_features_to_sample(sample::Sample, features::Dict{String,Float64})
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
            forest.featureValues[feature_name] = existing_feature_values
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
function create_tree(forest::Forest, feature_values::Dict{String,Array}, depth::UInt64)

    # Sanity check
    feature_values_len = length(feature_values)
    if feature_values_len <= 1
        return Nothing
    end

    # If we've exceeded the maximum desired depth, then stop.
    if (forest.subSamplingSize > 0) && (depth >= forest.subSamplingSize)
        return Nothing
    end

    # Randomly select a feature.
    selected_feature_index = rand(1:feature_values_len)
    all_feature_names = collect(keys(feature_values))
    selected_feature_name = all_feature_names[selected_feature_index]

    # Randomly select a split value.
    feature_value_set = feature_values[selected_feature_name]
    feature_value_set_len = length(feature_value_set)
    if feature_value_set_len <= 1
        return Nothing
    end
    split_value_index = rand(1:feature_value_set_len)
    split_value = feature_value_set[split_value_index]

    # Create a tree node to hold the split value.
    tree = Node(selected_feature_name, split_value, Nothing, Nothing)

    # Create two versions of the feature value set that we just used,
    # one for the left side of the tree and one for the right.
    temp_feature_values = feature_values

    # Create the left subtree.
    left_features = feature_value_set[1:split_value_index]
    temp_feature_values[selected_feature_name] = left_features
    tree.left = IsolationForest.create_tree(forest, temp_feature_values, depth + 1)

    # Create the right subtree.
    if split_value_index + 1 < feature_value_set_len
        right_features = feature_value_set[split_value_index + 1:feature_value_set_len]
        temp_feature_values[selected_feature_name] = right_features
        tree.right = IsolationForest.create_tree(forest, temp_feature_values, depth + 1)
    end

    return tree
end

# Creates a forest containing the number of trees specified to the constructor.
function create_forest(forest::Forest)
    for i = 1:forest.numTrees
        temp_feature_values = deepcopy(forest.featureValues)
        tree = create_tree(forest, temp_feature_values, UInt64(0))
        if tree != Nothing
            push!(forest.trees, tree)
        end
    end
end

# Scores the sample against the specified tree.
function score_sample_against_tree(tree::Node, sample::Sample)
    depth = 0.0
    current_node = tree

    while current_node != Nothing
        found_feature = false

        # Find the next feature in the sample.
        for current_feature in sample.features
            current_feature_name = current_feature.first

            # If the current node has the feature in question.
            if current_feature_name == current_node.featureName
                current_feature_value = current_feature.second
                if current_feature_value < current_node.splitValue
                    current_node = current_node.left
                else
                    current_node = current_node.right
                end

                depth = depth + 1.0
                found_feature = true
                break
            end
        end

        # If the tree contained a feature not in the sample then take
        # both sides of the tree and average the scores together.
        if found_feature == false
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
function H(i)
    return log(i) + 0.5772156649
end
function C(n)
    return 2 * H(n - 1) - (2 * (n - 1) / n)
end
function score_sample_against_forest_normalized(forest::Forest, sample::Sample)

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
        exponent = -1.0 * (avg_path_len / C(num_trees))
        score = 2 ^ exponent
    end

    return score
end

# Destroys a single tree.
function destroy_tree(tree::Node)
end

# Destroys the entire forest of trees.
function destroy()
    for tree in self.trees
        self.destroy_tree(tree)
    end
end

# Returns the specified node as a dictionary object.
function dump_node(node::Node)
    data = Dict()
    if node != Nothing
        data["Feature Name"] = node.featureName
        data["Split Value"] = node.splitValue
        if node.left == Nothing
            data["Left"] = Dict()
        else
            data["Left"] = dump_node(node.left)
        end
        if node.right == Nothing
            data["Right"] = Dict()
        else
            data["Right"] = dump_node(node.right)
        end
    end
    return data
end

# Loads/creates a node from a JSON object.
function load_node(data::Dict)
    try
        node = Node(data["Feature Name"], data["Split Value"])
        node.left = load_node(data["Left"])
        node.right = load_node(data["Right"])
        return node
    catch error
    end

    return Nothing
end

# Returns the specified tree as a dictionary object.
function dump_tree(tree::Node)
    data = dump_node(tree)
    return data
end

# Loads/creates a tree from a dictionary object.
function load_tree(forest::Forest, data::Dict)
    tree = load_node(data)
    if tree != Nothing
        push!(forest.trees, tree)
    end
end

# Returns the forest as a JSON object.
function dump(forest::Forest)
    data = Dict()
    data["Sub Sampling Size"] = forest.subSamplingSize
    data["Feature Values"] = forest.featureValues
    tree_data = []
    for tree in forest.trees
        push!(tree_data, dump_tree(tree))
    end
    data["Trees"] = tree_data
    return data
end

# Loads the forest from a JSON object.
function load(data::Dict)
    forest = Forest(0, 0, Dict(), [])
    forest.subSamplingSize = data["Sub Sampling Size"]
    forest.featureValues = data["Feature Values"]
    for tree_data in data["Trees"]
        load_tree(forest, tree_data)
    end
    forest.numTrees = length(forest.trees)
    return forest
end

end
