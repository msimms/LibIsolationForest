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
	let num_tests = 10;
	let mut forest = isolation_forest::Forest::new(10, 10);
    let mut rng = rand::thread_rng();
    let range1 = rand::distributions::Range::new(0.0, 25.0);
    let range2 = rand::distributions::Range::new(20.0, 45.0);

	// Training samples.
	for _i in 0..100
	{
		let mut sample = isolation_forest::Sample::new();
		let mut features = isolation_forest::FeatureList::new();

		let x = range1.ind_sample(&mut rng) as u64;
		let y = range1.ind_sample(&mut rng) as u64;

		features.push(isolation_forest::Feature::new("x", x));
		features.push(isolation_forest::Feature::new("y", y));

		sample.add_features(&mut features);
		forest.add_sample(sample);
	}

	// Create the isolation forest.
	forest.create();

	// Test samples (similar to training samples).
	println!("Test samples that are similar to the training set.");
	println!("--------------------------------------------------");
	let mut avg_normal_score = 0.0;
	for i in 0..num_tests
	{
		let mut sample = isolation_forest::Sample::new();
		let mut features = isolation_forest::FeatureList::new();

		let x = range1.ind_sample(&mut rng) as u64;
		let y = range1.ind_sample(&mut rng) as u64;

		features.push(isolation_forest::Feature::new("x", x));
		features.push(isolation_forest::Feature::new("y", y));
		sample.add_features(&mut features);

		// Run a test with the sample that doesn't contain outliers.
		let score = forest.score(&sample);
		avg_normal_score = avg_normal_score + score;
		println!("Normal test sample {}: {}", i, score);
	}
	avg_normal_score = avg_normal_score / num_tests as f64;
    println!("Average of normal test samples: {}.", avg_normal_score);

	// Outlier samples (different from training samples).
	println!("\nTest samples that are different from the training set.");
	println!("------------------------------------------------------");
	let mut avg_outlier_score = 0.0;
	for i in 0..num_tests
	{
		let mut sample = isolation_forest::Sample::new();
		let mut features = isolation_forest::FeatureList::new();

		let x = range2.ind_sample(&mut rng) as u64;
		let y = range2.ind_sample(&mut rng) as u64;

		features.push(isolation_forest::Feature::new("x", x));
		features.push(isolation_forest::Feature::new("y", y));
		sample.add_features(&mut features);

		// Run a test with the sample that contains outliers.
		let score = forest.score(&sample);
		avg_outlier_score = avg_outlier_score + score;
		println!("Outlier test sample {}: {}", i, score);
	}
	avg_outlier_score = avg_outlier_score / num_tests as f64;
    println!("Average of outlier test samples: {}.", avg_outlier_score);
}
