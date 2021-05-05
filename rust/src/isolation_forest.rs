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
 #![allow(dead_code)]

extern crate rand;
extern crate serde;
extern crate serde_json;

use std::collections::HashMap;
use rand::distributions::{Distribution, Uniform};
use self::serde::{
	ser::{SerializeStruct, Serializer}, Serialize, Deserialize
};

/// Each feature has a name and value.
#[derive(Clone, Serialize, Deserialize)]
pub struct Feature {
    pub name: String,
    value: u64,
}

impl Feature {
    pub fn new (name: &str, value: u64) -> Feature {
        Feature { name: name.to_string(), value: value }
    }
}

pub type FeatureList = Vec<Feature>;
pub type Uint64Vec = Vec<u64>;
pub type FeatureNameToValuesMap = HashMap<String, Uint64Vec>;

/// This class represents a sample.
/// Each sample has a name and list of features.
#[derive(Clone, Serialize, Deserialize)]
pub struct Sample {
    pub name: String,
    features: FeatureList,
}

impl Sample {
    pub fn new (sample_name: &str) -> Sample {
        Sample { name: sample_name.to_string(), features: Sample::create_feature_list() }
    }

    fn create_feature_list() -> FeatureList {
        let v: FeatureList = vec![];
        v
    }

    pub fn add_features(&mut self, features: &mut FeatureList) {
        self.features.append(features);
    }
}

/// Tree node, used internally.
#[derive(Serialize, Deserialize)]
struct Node {
    feature_name: String,
    split_value: u64,
    left: NodeLink,
    right: NodeLink,
}

impl Node {
    pub fn new (feature_name: &str, split_value: u64) -> Node {
        Node { feature_name: feature_name.to_string(), split_value: split_value, left: None, right: None }
    }
}

type NodeBox = Box<Node>;
type NodeLink = Option<Box<Node>>;
type NodeList = Vec<Box<Node>>;

/// Isolation Forest implementation.
pub struct Forest {
    feature_values: FeatureNameToValuesMap, // Lists each feature and maps it to all unique values in the training set
    trees: NodeList, // The decision trees that comprise the forest
    num_trees_to_create: u32, // The maximum number of trees to create
    sub_sampling_size: u32, // The maximum depth of a tree
    rng: rand::prelude::ThreadRng,
}

impl Serialize for Forest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("Forest", 3)?;
        s.serialize_field("Sub Sampling Size", &self.sub_sampling_size)?;
        s.serialize_field("Feature Values", &self.feature_values)?;
        s.serialize_field("Trees", &self.trees)?;
        s.end()
    }
}

impl Forest {
    pub fn new (num_trees_to_create: u32, sub_sampling_size: u32) -> Forest {
        Forest { num_trees_to_create: num_trees_to_create, sub_sampling_size: sub_sampling_size, trees: Forest::initialize_trees(), feature_values: Forest::create_feature_name_to_values_map(), rng: rand::thread_rng() }
    }

    fn initialize_trees() -> NodeList {
        let v: NodeList = vec![];
        v
    }

    fn create_feature_name_to_values_map() -> FeatureNameToValuesMap {
        let m = FeatureNameToValuesMap::new();
        m
    }

    pub fn add_sample(&mut self, sample: Sample) {
		// Add each of this sample's features to the list of known features
		// with the corresponding set of unique values.

        for feature in &sample.features {
            if self.feature_values.contains_key(&feature.name) {
                let mut feature_value_set = self.feature_values[&feature.name].clone();
                feature_value_set.push(feature.value);
                feature_value_set.sort_unstable();
                self.feature_values.insert(feature.name.clone(), feature_value_set);
            }
            else {
                let mut feature_value_set = Vec::new();
                feature_value_set.push(feature.value);
                self.feature_values.insert(feature.name.clone(), feature_value_set);
            }
        }
    }

    /// Creates and returns a single tree. As this is a recursive function, depth indicates the current depth of the recursion.
    fn create_tree(&mut self, feature_values: FeatureNameToValuesMap, depth: u32) -> NodeLink {
		// Sanity check.
        let feature_values_len = feature_values.len();
		if feature_values_len <= 1 {
			return None;
		}

		// If we've exceeded the maximum desired depth, then stop.
		if (self.sub_sampling_size > 0) && (depth >= self.sub_sampling_size) {
			return None;
		}

		// Randomly select a feature.
        let range = Uniform::from(0..feature_values_len);
		let selected_feature_index = range.sample(&mut self.rng) as usize;
        let selected_feature_name = feature_values.keys().nth(selected_feature_index);
        let unwrapped_feature_name = selected_feature_name.unwrap();

        // Randomly select a split value.
        let feature_value_set = &feature_values[unwrapped_feature_name];
        let feature_value_set_len = feature_value_set.len();
        if feature_value_set_len <= 1 {
            return None;
        }
        let range2 = Uniform::from(0..feature_value_set_len);
        let split_value_index = range2.sample(&mut self.rng) as usize;
        let split_value = feature_value_set[split_value_index];

        // Create a tree node to hold the split value.
        let mut tree_root = Node::new(unwrapped_feature_name, split_value);

        // Create two versions of the feature value set that we just used,
        // one for the left side of the tree and one for the right.
        let mut temp_feature_values = feature_values.clone();
        let (left_features, right_features) = feature_value_set.split_at(split_value_index);

        // Create the left subtree.
        temp_feature_values.insert(unwrapped_feature_name.to_string(), left_features.to_vec());
        tree_root.left = self.create_tree(temp_feature_values.clone(), depth + 1);

        // Create the right subtree.
        if right_features.len() > 0 {
            temp_feature_values.insert(unwrapped_feature_name.to_string(), right_features.to_vec());
            tree_root.right = self.create_tree(temp_feature_values.clone(), depth + 1);
        }

        let tree = Some(Box::new(tree_root));
        tree
    }

    /// Creates a forest containing the number of trees specified to the constructor.
    pub fn create(&mut self) {
    	for _i in 0..self.num_trees_to_create {
            let temp_feature_values = self.feature_values.clone();
            let tree = self.create_tree(temp_feature_values, 0);
            if tree.is_some() {
                self.trees.push(tree.unwrap());
            }
        }
    }

    /// Scores the sample against the specified tree.
    fn score_tree(&self, sample: &Sample, tree: &NodeBox) -> f64 {
        let mut depth = 0.0;
        let mut current_node = tree;
        let mut done = false;

        while !done {
            let mut found_feature = false;

            for current_feature in &sample.features {

                // If the current node has the feature in question.
                if current_feature.name == current_node.feature_name {
					if current_feature.value < current_node.split_value {
                        match current_node.left {
                            None => {
                                done = true;
                            }
                            Some(ref next_node) => {
                                current_node = next_node;
                            }
                        }
                    } else {
                        match current_node.right {
                            None => {
                                done = true;
                            }
                            Some(ref next_node) => {
                                current_node = next_node;
                            }
                        }
                    }

                    depth = depth + 1.0;
                    found_feature = true;
                }
            }

			// If the tree contained a feature not in the sample then take
			// both sides of the tree and average the scores together.
			if found_feature == false {
                let left_tree = &current_node.left;
                let right_tree = &current_node.right;
                let mut left_depth = depth;
                let mut right_depth = depth;

                match left_tree {
                    &None => {
                    }
                    &Some(ref left_tree) => {
        			    left_depth = left_depth + self.score_tree(sample, left_tree);
                    }
                }
                match right_tree {
                    &None => {
                    }
                    &Some(ref right_tree) => {
        			    right_depth = right_depth + self.score_tree(sample, right_tree);
                    }
                }
				depth = (left_depth + right_depth) / 2.0;
                return depth;
            }
        }
        depth
    }

    /// Scores the sample against the entire forest of trees. Result is the average path length.
    pub fn score(&self, sample: &Sample) -> f64 {
        let mut score = 0.0;

        if self.trees.len() > 0 {
            for tree in &self.trees {
                score += self.score_tree(sample, tree) as f64;
            }
            score /= self.trees.len() as f64;
        }
        score
    }

    fn h(&self, i: usize) -> f64 {
        let result = (i as f64).ln() + 0.5772156649;
        result
    }

    fn c(&self, n: usize) -> f64 {
        let result = 2.0 * self.h(n - 1) - (2 * (n - 1) / n) as f64;
        result
    }

    /// Scores the sample against the entire forest of trees. Result is normalized so that values
    /// close to 1 indicate anomalies and values close to zero indicate normal values."""
    pub fn normalized_score(&self, sample: &Sample) -> f64 {
        let mut score = 0.0;
        let mut avg_path_len = 0.0;
        let num_trees = self.trees.len();

        // Compute the average path length for all valid trees.
        if num_trees > 0 {
            for tree in &self.trees {
                avg_path_len += self.score_tree(sample, tree) as f64;
            }
            avg_path_len /= self.trees.len() as f64;

            let exponent = -1.0 * (avg_path_len / self.c(num_trees));
            let x = 2.0_f64;
            score = x.powf(exponent);
        }

        // Normalize, per the original paper.
        score
    }

    pub fn dump(&self) -> String {
        let json_str = serde_json::to_string(&self).unwrap();
        json_str
    }

    pub fn load(&self, json_str: &String) {
        let obj = serde_json::from_str(json_str).unwrap();
        obj
    }
}
