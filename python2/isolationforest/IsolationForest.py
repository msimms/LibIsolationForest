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

import copy
import math
import random

class Node(object):
    """Tree node, used internally."""

    def __init__(self, feature_name, split_value):
        self.feature_name = feature_name
        self.split_value = split_value
        self.left = None
        self.right = None

class Sample(object):
    """This class represents a sample. Each sample has a name and list of features."""

    def __init__(self, name):
        self.name = name
        self.features = []

    def add_features(self, features):
        for feature in features:
            self.add_feature(feature)

    def add_feature(self, feature):
        self.features.append(feature)

class Forest(object):
    """Isolation Forest implementation."""
 
    def __init__(self, num_trees, sub_sampling_size):
        self.num_trees = num_trees
        self.sub_sampling_size = sub_sampling_size
        self.feature_values = {}
        self.trees = []

    def add_sample(self, sample):
        """Adds each of the sample's features to the list of known features 
           with the corresponding set of unique values."""

        # We don't store the sample directly, just the features.
        for feature in sample.features:
            feature_name = feature.keys()[0]
            feature_value = feature.values()[0]
            if feature_name in self.feature_values:
                self.feature_values[feature_name].append(feature_value)
                self.feature_values[feature_name].sort()
            else:
                feature_value_set = []
                feature_value_set.append(feature_value)
                self.feature_values[feature_name] = feature_value_set

    def create_tree(self, feature_values, depth):
        """Creates and returns a single tree. As this is a recursive function, depth indicates the current depth of the recursion."""

        # Sanity check
        feature_values_len = len(feature_values)
        if feature_values_len <= 1:
            return None

		# If we've exceeded the maximum desired depth, then stop.
        if (self.sub_sampling_size > 0) and (depth >= self.sub_sampling_size):
            return None

        # Randomly select a feature.
        selected_feature_index = random.randint(0, feature_values_len - 1)
        selected_feature_name = feature_values.keys()[selected_feature_index]

        # Randomly select a split value.
        feature_value_set = feature_values[selected_feature_name]
        feature_value_set_len = len(feature_value_set)
        if feature_value_set_len <= 1:
            return None
        split_value_index = random.randint(0, feature_value_set_len - 1)
        split_value = feature_value_set[split_value_index]

        # Create a tree node to hold the split value.
        tree = Node(selected_feature_name, split_value)

        # Create two versions of the feature value set that we just used,
        # one for the left side of the tree and one for the right.
        temp_feature_values = feature_values

        # Create the left subtree.
        left_features = feature_value_set[:split_value_index]
        temp_feature_values[selected_feature_name] = left_features
        tree.left = self.create_tree(temp_feature_values, depth + 1)

        # Create the right subtree.
        if split_value_index + 1 < feature_value_set_len:
            right_features = feature_value_set[split_value_index + 1:]
            temp_feature_values[selected_feature_name] = right_features
            tree.right = self.create_tree(temp_feature_values, depth + 1)

        return tree

    def create(self):
        """Creates a forest containing the number of trees specified to the constructor."""
        for _ in range(0, self.num_trees):
            temp_feature_values = copy.deepcopy(self.feature_values)
            tree = self.create_tree(temp_feature_values, 0)
            self.trees.append(tree)

    def score_tree(self, sample, tree):
        """Scores the sample against the specified tree."""
        depth = 0.0
        current_node = tree
        while current_node is not None:
            found_feature = False

            # Find the next feature in the sample.
            for current_feature in sample.features:
                current_feature_name = current_feature.keys()[0]

                # If the current node has the feature in question.
                if current_feature_name == current_node.feature_name:
                    current_feature_value = current_feature.values()[0]
                    if current_feature_value < current_node.split_value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
                    depth = depth + 1.0
                    found_feature = True
                    break

            # If the tree contained a feature not in the sample then take
            # both sides of the tree and average the scores together.
            if not found_feature:
                left_depth = depth + self.score_tree(sample, current_node.left)
                right_depth = depth + self.score_tree(sample, current_node.right)
                return (left_depth + right_depth) / 2.0
        return depth

    def score(self, sample):
        """Scores the sample against the entire forest of trees. Result is the average path length."""
        num_trees = 0.0
        avg_path_len = 0.0
        if self.trees is not None:
            for tree in self.trees:
                path_len = self.score_tree(sample, tree)
                if path_len > 0:
                    avg_path_len = avg_path_len + path_len
                    num_trees = num_trees + 1.0
            if num_trees > 0.0:
                avg_path_len = avg_path_len / num_trees
        return avg_path_len

    def normalized_score(self, sample):
        """ Scores the sample against the entire forest of trees. Result is normalized so that values
            close to 1 indicate anomalies and values close to zero indicate normal values."""
        def H(i):
            return math.log(i) + 0.5772156649
        def C(n):
            return 2 * H(n - 1) - (2 * (n - 1) / n)

        # Compute the average path length for all valid trees.
        num_trees = 0.0
        avg_path_len = 0.0
        if self.trees is not None:
            for tree in self.trees:
                path_len = self.score_tree(sample, tree)
                if path_len > 0:
                    avg_path_len = avg_path_len + path_len
                    num_trees = num_trees + 1.0
            if num_trees > 0.0:
                avg_path_len = avg_path_len / num_trees
        
        # Normalize, per the original paper.
        score = 0.0
        if num_trees > 1.0:
            exponent = -1.0 * (avg_path_len / C(num_trees))
            score = pow(2, exponent)
        return score

    def destroy_tree(self, tree):
        """Destroys a single tree."""
        pass

    def destroy(self):
        """Destroys the entire forest of trees."""
        for tree in self.trees:
            self.destroy_tree(tree)

    def dump_node(self, node):
        """Returns the specified node as a dictionary object."""
        data = {}
        if node is not None:
            data["Feature Name"] = node.feature_name
            data["Split Value"] = node.split_value
            data["Left"] = self.dump_node(node.left)
            data["Right"] = self.dump_node(node.right)
        return data

    def dump_tree(self, tree):
        """Returns the specified tree as a dictionary object."""
        data = self.dump_node(tree)
        return data

    def dump(self):
        """Returns the forest as a JSON object."""
        data = {}
        tree_index = 0
        for tree in self.trees:
            data["Tree " + str(tree_index)] = self.dump_tree(tree)
            tree_index = tree_index + 1
        return data

    def load(self, data):
        """Loads the forest from the JSON object."""
        pass
