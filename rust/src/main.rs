//	MIT License
//
//  Copyright © 2018 Michael J Simms. All rights reserved.
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

mod isolation_forest;

extern crate rand;
use rand::distributions::IndependentSample;

fn main()
{
	let mut forest = isolation_forest::Forest::new(10, 10);
    let mut rng = rand::thread_rng();
    let range = rand::distributions::Range::new(0.0, 100.0);

	// Training samples.
	for _i in 0..100
	{
		let mut sample = isolation_forest::Sample::new("");
		let mut features = isolation_forest::FeatureList::new();

		let x = 0.3 * (range.ind_sample(&mut rng));
		let y = 0.3 * (range.ind_sample(&mut rng));
		let x = x as u64;
		let y = y as u64;

		features.push(isolation_forest::Feature::new("x", x));
		features.push(isolation_forest::Feature::new("y", y));

		sample.add_features(&mut features);
		forest.add_sample(sample);
	}

	// Create the isolation forest.
	forest.create();

	// Test samples (similar to training samples).
	println!("Test samples that are similar to the training set.\n");
	println!("--------------------------------------------------\n");
	for i in 0..10
	{
		let mut sample = isolation_forest::Sample::new("");
		let mut features = isolation_forest::FeatureList::new();

		let x = 0.3 * (range.ind_sample(&mut rng));
		let y = 0.3 * (range.ind_sample(&mut rng));
		let x = x as u64;
		let y = y as u64;

		features.push(isolation_forest::Feature::new("x", x));
		features.push(isolation_forest::Feature::new("y", y));
		sample.add_features(&mut features);

		// Run a test with the sample that doesn't contain outliers.
		let score = forest.score(&sample);
		println!("Normal test sample {}: {}\n", i, score);
	}

	// Outlier samples (different from training samples).
	println!("Test samples that are different from the training set.\n");
	println!("------------------------------------------------------\n");
	for i in 0..10
	{
		let mut sample = isolation_forest::Sample::new("");
		let mut features = isolation_forest::FeatureList::new();

		let x = 25.0 + (0.5 * (range.ind_sample(&mut rng)));
		let y = 25.0 + (0.5 * (range.ind_sample(&mut rng)));
		let x = x as u64;
		let y = y as u64;

		features.push(isolation_forest::Feature::new("x", x));
		features.push(isolation_forest::Feature::new("y", y));
		sample.add_features(&mut features);

		// Run a test with the sample that contains outliers.
		let score = forest.score(&sample);
		println!("Outlier test sample {}: {}\n", i, score);
	}
}
