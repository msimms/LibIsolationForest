//	MIT License
//
//  Copyright © 2017 Michael J Simms. All rights reserved.
//
//	Permission is hereby granted, free of charge, to any person obtaining a copy
//	of this software and associated documentation files (the "Software"), to deal
//	in the Software without restriction, including without limitation the rights
//	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//	copies of the Software, and to permit persons to whom the Software is
//	furnished to do so, subject to the following conditions:
//
//	The above copyright notice and this permission notice shall be included in all
//	copies or substantial portions of the Software.
//
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//	SOFTWARE.

use std::collections::HashSet;
use std::collections::HashMap;

/// Each feature has a name and value.
pub struct Feature {
    name: String,
    value: u64,
}

impl Feature {
    pub fn new (name: &str, value: u64) -> Feature {
        Feature { name: name.to_string(), value: value }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn get_name(&self) -> String {
        self.name
    }

    pub fn set_value(&mut self, value: u64) {
        self.value = value;
    }

    pub fn get_value(&self) -> u64 {
        self.value
    }
}

pub type FeatureList = Vec<Feature>;
pub type Uint64Set = HashSet<u64>;
pub type FeatureNameToValuesMap<'a> = HashMap<&'a str, Uint64Set>;

/// This class represents a sample.
/// Each sample has a name and list of features.
pub struct Sample {
    name: String,
    features: FeatureList,
}

impl Sample {
    pub fn new (name: &str) -> Sample {
        Sample { name: name.to_string(), features: Sample::create_feature_list() }
    }

    fn create_feature_list() -> FeatureList {
        let mut v: FeatureList = vec![];
        v
    }

    pub fn add_features(&mut self, features: &mut FeatureList) {
        self.features.append(features);
    }

    pub fn add_feature(&mut self, feature: Feature) {
        self.features.push(feature);
    }

    pub fn features(&self) -> FeatureList {
        self.features
    }
}

pub type SampleList = Vec<Sample>;

/// Tree node, used internally.
struct Node {
    feature_name: String,
    left: NodeLink,
    right: NodeLink,
}

impl Node {
    pub fn new (feature_name: &str) -> Node {
        Node { feature_name: feature_name.to_string(), left: None, right: None }
    }

    fn get_feature_name(&self) -> String {
        self.feature_name
    }

	fn set_left_subtree(&mut self, subtree: NodeLink) {
		self.left = subtree;
	}

    fn get_left_subtree(&self) -> NodeLink {
        self.left
    }

	fn set_right_subtree(&mut self, subtree: NodeLink) {
		self.right = subtree;
	}

    fn get_right_subtree(&self) -> NodeLink {
        self.right
    }
}

pub type NodeLink = Option<Box<Node>>;
pub type NodeList = Vec<Box<Node>>;

/// Isolation Forest implementation.
pub struct Forest<'a> {
    feature_values: FeatureNameToValuesMap<'a>, // Lists each feature and maps it to all unique values in the training set
    trees: NodeList, // The decision trees that comprise the forest
    num_trees_to_create: u32, // The maximum number of trees to create
    sub_sampling_size: u32, // The maximum depth of a tree
}

impl<'a> Forest<'a> {
    pub fn new (num_trees_to_create: u32, sub_sampling_size: u32) -> Forest<'a> {
        Forest { num_trees_to_create: num_trees_to_create, sub_sampling_size: sub_sampling_size, trees: Forest::initialize_trees(), feature_values: Forest::create_feature_name_to_values_map() }
    }

    fn initialize_trees() -> NodeList {
        let mut v: NodeList = vec![];
        v
    }

    fn create_feature_name_to_values_map() -> FeatureNameToValuesMap<'a> {
        let mut m = HashMap::new();
        m
    }

    pub fn add_sample(&self, sample: Sample) {
		// Add each of this sample's features to the list of known features
		// with the corresponding set of unique values.
		let mut features = sample.features();
        for feature in &features {
            let mut found_feature = false;
        }
    }

    pub fn create_tree(&self, feature_values: FeatureNameToValuesMap<'a>, depth: u32) -> NodeLink {
		// Sanity check.
		if feature_values.len() <= 1 {
			return None;
		}

		// If we've exceeded the maximum desired depth, then stop.
		if (self.sub_sampling_size > 0) && (depth >= self.sub_sampling_size) {
			return None;
		}

		// Randomly select a feature.
		let mut selected_feature_index = 0;
		if feature_values.len() > 1 {
			return None;
		}

		// Get the value list to split on.

		// Randomly select a split value.
		let mut split_value_index = 0;

		// Create a tree node to hold the split value.
        let mut tree = Some(Box::new(Node::new("")));

        tree
    }

    pub fn create(&self) {
    	for _i in 0..self.num_trees_to_create {
            let tree = self.create_tree(self.feature_values, 0);
            if !tree.is_none() {
            }
        }
    }

    fn score_tree(&self, sample: Sample, tree: NodeLink) -> f64 {
        let mut depth = 0.0;
        let mut current_node = tree;
        let mut done = false;

        while !done {
            let mut found_feature = false;
            let current_node_deref = current_node;
            let node_feature_name = current_node_deref.get_feature_name();
            let features = sample.features();

            for feature in features {
                let feature_name = feature.get_name();

                if feature_name == node_feature_name {
					if feature.get_value() < current_node_deref.split_value() {
						current_node = current_node_deref.get_left_subtree();
                    } else {
						current_node = current_node_deref.get_right_subtree();
                    }
                    found_feature = false;
                }
            }
        }
        depth
    }

    pub fn score(&self, sample: Sample) -> f64 {
        let mut score = 0.0;

        if self.trees.len() > 0 {
            for tree in self.trees {
                score += self.score_tree(sample, Some(tree));
            }
            score /= self.trees.len() as f64;
        }
        score
    }
}
