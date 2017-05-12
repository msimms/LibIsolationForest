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
	
IsolationForest::IsolationForest() :
	m_numTreesToCreate(0),
	m_subSamplingSize(0)
{
}

IsolationForest::IsolationForest(uint32_t numTrees, uint32_t subSamplingSize) :
	m_numTreesToCreate(numTrees),
	m_subSamplingSize(subSamplingSize)
{
}

IsolationForest::~IsolationForest()
{
	Destroy();
}

void IsolationForest::Fit(const FeatureList& nodes)
{
	for (size_t i = 0; i < m_numTreesToCreate; ++i)
	{
		FeatureList::const_iterator iter = nodes.begin();
		while (iter != nodes.end())
		{
			++iter;
		}
	}
}

void IsolationForest::Predict(const FeatureList& nodes)
{

}

FeatureTypePtr IsolationForest::Insert(FeatureTypePtr& root, const FeatureTypePtr& node)
{
	if (!root)
		root = node;
	else if (node < root)
		root->left = Insert(root->left, node);
	else
		root->right = Insert(root->right, node);
	return root;
}

void IsolationForest::Destroy()
{
}

};
