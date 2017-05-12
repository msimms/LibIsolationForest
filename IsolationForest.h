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

#pragma once

#include <stdint.h>
#include <string>
#include <vector>

namespace IsolationForest
{
	
template <class K, class V>
class Feature
{
public:
	Feature() { left = right = NULL; };
	virtual ~Feature() {};
	
	virtual bool operator <(Feature const& b) = 0;
	
	virtual const K& Key() const = 0;
	virtual const V& Value() const = 0;
	
	Feature* left;
	Feature* right;
};

typedef Feature<class K, class V> FeatureType;
typedef FeatureType* FeatureTypePtr;
typedef std::vector<FeatureTypePtr> FeatureList;

class Sample
{
public:
	Sample() {};
	virtual ~Sample() {};

	std::string name;

};

class IsolationForestRandomizer
{
public:
	IsolationForestRandomizer() {} ;
	virtual ~IsolationForestRandomizer() {};
};

class IsolationForest
{
public:
	IsolationForest();
	IsolationForest(uint32_t numTrees, uint32_t subSamplingSize);
	virtual ~IsolationForest();
	
	void Fit(const FeatureList& nodes);
	void Predict(const FeatureList& nodes);

private:
	std::vector<FeatureTypePtr> m_trees;
	uint32_t m_numTreesToCreate;
	uint32_t m_subSamplingSize;
	
	FeatureTypePtr Insert(FeatureTypePtr& root, const FeatureTypePtr& node);
	void Destroy();
};

};
