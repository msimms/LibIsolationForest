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

namespace IsolationForest
{
	Node::Node() :
		m_left(NULL),
		m_right(NULL),
		m_splitValue(0)
	{
	}

	Node::Node(uint64_t splitValue) :
		m_left(NULL),
		m_right(NULL),
		m_splitValue(splitValue)
	{
	}

	Node::~Node()
	{
		if (m_left)
		{
			delete m_left;
			m_left = NULL;
		}
		if (m_right)
		{
			delete m_right;
			m_right = NULL;
		}
	}

	Forest::Forest() :
		m_randomizer(new Randomizer),
		m_numTreesToCreate(10),
		m_subSamplingSize(0)
	{
	}

	Forest::Forest(uint32_t numTrees, uint32_t subSamplingSize) :
		m_randomizer(new Randomizer),
		m_numTreesToCreate(numTrees),
		m_subSamplingSize(subSamplingSize)
	{
	}

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

	void Forest::AddSample(const Sample& sample)
	{
		// Update the min and max values for each feature of this sample.
		const FeaturePtrList& features = sample.Features();
		FeaturePtrList::const_iterator featureIter = features.begin();
		while (featureIter != features.end())
		{
			const FeaturePtr feature = (*featureIter);
			const std::string& featureName = feature->Name();
			uint64_t featureValue = feature->Value();

			// Add the feature to the list if this is the first time we're seeing it.
			// Otherwise, just update the min and max values.
			if (m_featureMinMax.count(featureName) == 0)
			{
				Uint64Pair featureValuePair = std::make_pair(featureValue, featureValue);
				m_featureMinMax.insert(std::make_pair(featureName, featureValuePair));
			}
			else
			{
				Uint64Pair& featureValuePair = m_featureMinMax.at(featureName);
				if (featureValuePair.first < featureValue)
					featureValuePair.first = featureValue;
				if (featureValuePair.second > featureValue)
					featureValuePair.second = featureValue;
			}
			++featureIter;
		}
	}

	NodePtr Forest::CreateTree()
	{
		// Sanity check.
		if (m_featureMinMax.size() <= 1)
		{
			return NULL;
		}

		NodePtr tree = NULL;
		
		// Randomly select a feature.
		size_t selectedFeatureIndex = (size_t)m_randomizer->RandUInt64(0, m_featureMinMax.size() - 1);
		std::map<std::string, Uint64Pair>::const_iterator featureIter = m_featureMinMax.begin();
		std::advance(featureIter, selectedFeatureIndex);

		// Randomly select a split value, somewhere between the min and max values.
		const Uint64Pair& minMax = (*featureIter).second;
		size_t splitValue = (size_t)m_randomizer->RandUInt64(minMax.first, minMax.second);

		// Create a tree node to hold the split value.
		NodePtr node = new Node(splitValue);

		// If this is the first node we've created then it is, by definition, the root of the tree.
		if (!tree)
		{
			tree = node;
		}

		return tree;
	}

	void Forest::Create()
	{
		for (size_t i = 0; i < m_numTreesToCreate; ++i)
		{
			NodePtr tree = CreateTree();
			if (tree)
			{
				m_trees.push_back(tree);
			}
		}
	}

	void Forest::Predict(const Sample& sample)
	{
		const FeaturePtrList& features = sample.Features();
		FeaturePtrList::const_iterator featureIter = features.begin();
		while (featureIter != features.end())
		{
			++featureIter;
		}
	}

	void Forest::DestroyTree(NodePtr tree)
	{
		if (tree)
		{
			delete tree;
		}
	}

	void Forest::Destroy()
	{
		std::vector<NodePtr>::iterator iter = m_trees.begin();
		while (iter != m_trees.end())
		{
			DestroyTree((*iter));
			++iter;
		}
		m_trees.clear();
	}

	void Forest::DestroyRandomizer()
	{
		if (m_randomizer)
		{
			delete m_randomizer;
			m_randomizer = NULL;
		}
	}
};
