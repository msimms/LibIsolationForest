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
#include <stdlib.h>

using namespace IsolationForest;

int main(int argc, const char * argv[])
{
	std::vector<FeaturePtr> training;
	std::vector<FeaturePtr> test;
	std::vector<FeaturePtr> outliers;

	srand((unsigned int)time(NULL));

	for (size_t i = 0; i < 100; ++i)
	{
		uint32_t x = 0.3 * rand();
		training.push_back(new Feature("foo", x));
	}

	for (size_t i = 0; i < 20; ++i)
	{
		uint32_t x = 0.3 * rand();
		test.push_back(new Feature("foo", x));
	}

	return 0;
}
