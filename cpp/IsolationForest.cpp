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

#include "IsolationForest.h"
#include <math.h>

namespace IsolationForest
{
	/// Constructor.
	Node::Node() :
		m_splitValue(0),
		m_left(NULL),
		m_right(NULL)
	{
	}

	/// Constructor.
	Node::Node(const std::string& featureName, uint64_t splitValue) :
		m_featureName(featureName),
		m_splitValue(splitValue),
		m_left(NULL),
		m_right(NULL)
	{
	}

	/// Destructor.
	Node::~Node()
	{
		DestroyLeftSubtree();
		DestroyRightSubtree();
	}

	void Node::SetLeftSubTree(Node* subtree)
	{
		DestroyLeftSubtree();
		m_left = subtree;
	}

	void Node::SetRightSubTree(Node* subtree)
	{
		DestroyRightSubtree();
		m_right = subtree;
	}

	void Node::DestroyLeftSubtree()
	{
		if (m_left)
		{
			delete m_left;
			m_left = NULL;
		}
	}

	void Node::DestroyRightSubtree()
	{
		if (m_right)
		{
			delete m_right;
			m_right = NULL;
		}
	}

	/// Returns the node as a JSON string.
	std::string Node::Dump() const
	{
		std::string data = "{";

		data.append("'Feature Name': '");
		data.append(this->m_featureName);
		data.append("', 'Split Value': ");
		data.append(std::to_string(m_splitValue));
		data.append(", 'Left': ");
		if (this->m_left)
			data.append(this->m_left->Dump());
		else
			data.append("{}");
		data.append(", 'Right': ");
		if (this->m_right)
			data.append(this->m_right->Dump());
		else
			data.append("{}");
		data.append("}");
		return data;
	}

	/// Constructor.
	Forest::Forest() :
		m_randomizer(new Randomizer()),
		m_numTreesToCreate(10),
		m_subSamplingSize(0)
	{
	}

	/// Constructor.
	Forest::Forest(uint32_t numTrees, uint32_t subSamplingSize) :
		m_randomizer(new Randomizer()),
		m_numTreesToCreate(numTrees),
		m_subSamplingSize(subSamplingSize)
	{
	}

	/// Destructor.
	Forest::~Forest()
	{
		DestroyRandomizer();
		Destroy();
	}

	void Forest::SetRandomizer(Randomizer* newRandomizer)
	{
		DestroyRandomizer();
		m_randomizer = newRandomizer;
	}

	/// Adds each of the sample's features to the list of known features
	/// with the corresponding set of unique values.
	void Forest::AddSample(const Sample& sample)
	{
        // We don't store the sample directly, just the features.
		const FeaturePtrList& features = sample.Features();
		FeaturePtrList::const_iterator featureIter = features.begin();
		while (featureIter != features.end())
		{
			const FeaturePtr feature = (*featureIter);
			const std::string& featureName = feature->Name();
			uint64_t featureValue = feature->Value();

			// Either create or update the feature values count.
			if (m_featureValues.count(featureName) == 0)
			{
				Uint64Set featureValueSet;
				featureValueSet.insert(featureValue);
				m_featureValues.insert(std::make_pair(featureName, featureValueSet));
			}
			else
			{
				Uint64Set& featureValueSet = m_featureValues.at(featureName);
				featureValueSet.insert(featureValue);
			}

			++featureIter;
		}
	}

	/// Creates and returns a single tree. As this is a recursive function,
	/// depth indicates the current depth of the recursion.
	NodePtr Forest::CreateTree(const FeatureNameToValuesMap& featureValues, size_t depth)
	{
		// Sanity check.
		size_t featureValuesLen = featureValues.size();
		if (featureValuesLen <= 1)
		{
			return NULL;
		}

		// If we've exceeded the maximum desired depth, then stop.
		if ((m_subSamplingSize > 0) && (depth >= m_subSamplingSize))
		{
			return NULL;
		}

		// Randomly select a feature.
		size_t selectedFeatureIndex = (size_t)m_randomizer->RandUInt64(0, featureValuesLen - 1);
		FeatureNameToValuesMap::const_iterator featureIter = featureValues.begin();
		std::advance(featureIter, selectedFeatureIndex);
		const std::string& selectedFeatureName = (*featureIter).first;

		// Get the value list to split on.
		const Uint64Set& featureValueSet = (*featureIter).second;
		if (featureValueSet.size() == 0)
		{
			return NULL;
		}

		// Randomly select a split value.
		size_t splitValueIndex = 0;
		if (featureValueSet.size() > 1)
		{
			splitValueIndex = (size_t)m_randomizer->RandUInt64(0, featureValueSet.size() - 1);
		}
		Uint64Set::const_iterator splitValueIter = featureValueSet.begin();
		std::advance(splitValueIter, splitValueIndex);
		uint64_t splitValue = (*splitValueIter);

		// Create a tree node to hold the split value.
		NodePtr tree = new Node(selectedFeatureName, splitValue);
		if (tree)
		{
			// Create two versions of the feature value set that we just used,
			// one for the left side of the tree and one for the right.
			FeatureNameToValuesMap tempFeatureValues = featureValues;

			// Create the left subtree.
			Uint64Set leftFeatureValueSet = featureValueSet;
			splitValueIter = leftFeatureValueSet.begin();
			std::advance(splitValueIter, splitValueIndex);
			leftFeatureValueSet.erase(splitValueIter, leftFeatureValueSet.end());
			tempFeatureValues[selectedFeatureName] = leftFeatureValueSet;
			tree->SetLeftSubTree(CreateTree(tempFeatureValues, depth + 1));

			// Create the right subtree.
			if (splitValueIndex < featureValueSet.size() - 1)
			{
				Uint64Set rightFeatureValueSet = featureValueSet;
				splitValueIter = rightFeatureValueSet.begin();
				std::advance(splitValueIter, splitValueIndex + 1);
				rightFeatureValueSet.erase(rightFeatureValueSet.begin(), splitValueIter);
				tempFeatureValues[selectedFeatureName] = rightFeatureValueSet;
				tree->SetRightSubTree(CreateTree(tempFeatureValues, depth + 1));
			}
		}

		return tree;
	}

	/// Creates a forest containing the number of trees specified to the constructor.
	void Forest::Create()
	{
		m_trees.reserve(m_numTreesToCreate);

		for (size_t i = 0; i < m_numTreesToCreate; ++i)
		{
			NodePtr tree = CreateTree(m_featureValues, 0);
			if (tree)
			{
				m_trees.push_back(tree);
			}
		}
	}

    /// Scores the sample against the specified tree.
	double Forest::Score(const Sample& sample, const NodePtr tree)
	{
		double depth = (double)0.0;

		const FeaturePtrList& features = sample.Features();

		NodePtr currentNode = tree;
		while (currentNode)
		{
			bool foundFeature = false;

			// Find the next feature in the sample.
			FeaturePtrList::const_iterator featureIter = features.begin();
			while (featureIter != features.end() && !foundFeature)
			{
				const FeaturePtr currentFeature = (*featureIter);
				if (currentFeature->Name().compare(currentNode->FeatureName()) == 0)
				{
					if (currentFeature->Value() < currentNode->SplitValue())
					{
						currentNode = currentNode->Left();
					}
					else
					{
						currentNode = currentNode->Right();
					}
					++depth;
					foundFeature = true;
				}
				++featureIter;
			}

			// If the tree contained a feature not in the sample then take
			// both sides of the tree and average the scores together.
			if (!foundFeature)
			{
				double leftDepth = depth + Score(sample, currentNode->Left());
				double rightDepth = depth + Score(sample, currentNode->Right());
				return (leftDepth + rightDepth) / (double)2.0;
			}
		}
		return depth;
	}

	/// Scores the sample against the entire forest of trees. Result is the average path length.
	double Forest::Score(const Sample& sample)
	{
		double avgPathLen = (double)0.0;
		
		if (m_trees.size() > 0)
		{
			NodePtrList::const_iterator treeIter = m_trees.begin();
			while (treeIter != m_trees.end())
			{
				avgPathLen += (double)Score(sample, (*treeIter));
				++treeIter;
			}
			avgPathLen /= (double)m_trees.size();
		}
		return avgPathLen;
	}

	#define H(i) (log(i) + 0.5772156649)
	#define C(n) (2 * H(n - 1) - (2 * (n - 1) / n))

	/// Scores the sample against the entire forest of trees. Result is normalized so that values
    /// close to 1 indicate anomalies and values close to zero indicate normal values.
	double Forest::NormalizedScore(const Sample& sample)
    {
		double score = (double)0.0;
		size_t numTrees = m_trees.size();

		if (numTrees > 0)
		{
			double avgPathLen = (double)0.0;

			NodePtrList::const_iterator treeIter = m_trees.begin();
			while (treeIter != m_trees.end())
			{
				avgPathLen += (double)Score(sample, (*treeIter));
				++treeIter;
			}
			avgPathLen /= (double)numTrees;

			if (numTrees > 1)
			{
				double exponent = -1.0 * (avgPathLen / C(numTrees));
				score = pow(2, exponent);
			}
		}
		return score;
    }

	/// Destroys the entire forest of trees.
	void Forest::Destroy()
	{
		std::vector<NodePtr>::iterator iter = m_trees.begin();
		while (iter != m_trees.end())
		{
			NodePtr tree = (*iter);
			if (tree)
			{
				delete tree;
			}
			++iter;
		}
		m_trees.clear();
	}

	/// Frees the custom randomizer object (if any).
	void Forest::DestroyRandomizer()
	{
		if (m_randomizer)
		{
			delete m_randomizer;
			m_randomizer = NULL;
		}
	}

	/// Returns the forest as a JSON object.
	std::string Forest::Dump() const
	{
		std::string data = "{";
		auto featureValuesIter = m_featureValues.begin();
		auto treeIter = m_trees.begin();

		data.append("'Sub Sampling Size': ");
		data.append(std::to_string(this->m_subSamplingSize));
		data.append(", 'Feature Values': [");
		while (featureValuesIter != m_featureValues.end())
		{
			data.append("'");
			data.append((*featureValuesIter).first);
			data.append("': [");

			auto valuesIter = (*featureValuesIter).second.begin();
			while (valuesIter != (*featureValuesIter).second.end())
			{
				data.append(std::to_string(*valuesIter));
				++valuesIter;

				// Last item?
				if (valuesIter != (*featureValuesIter).second.end())
					data.append(", ");
			}
			++featureValuesIter;

			data.append("]");

			// Last item?
			if (featureValuesIter != m_featureValues.end())
				data.append(", ");
		}
		data.append("], 'Trees': [");
		while (treeIter != m_trees.end())
		{
			data.append((*treeIter)->Dump());
			++treeIter;
			
			// Last item?
			if (treeIter != m_trees.end())
				data.append(", ");
		}
		data.append("]");
		data.append("}");
		return data;
	}
}
