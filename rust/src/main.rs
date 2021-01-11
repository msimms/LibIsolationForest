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
use rand::distributions::{Distribution, Uniform};
use std::env;

fn main()
{
    let args: Vec<String> = env::args().collect();
	let num_tests = 10;
	let num_training_samples = 10;
	let mut forest = isolation_forest::Forest::new(10, 10);
    let mut rng = rand::thread_rng();
    let range1 = Uniform::from(0..25);
    let range2 = Uniform::from(20..45);
    let mut dump = false;

    for arg in args
    {
        if arg == "--dump" {
            dump = true;
        }
    }

	// Training samples.
	for _i in 0..num_training_samples
	{
		let mut sample = isolation_forest::Sample::new();
		let mut features = isolation_forest::FeatureList::new();

		let x = range1.sample(&mut rng) as u64;
		let y = range1.sample(&mut rng) as u64;

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
	let mut avg_control_score = 0.0;
	let mut avg_control_normalized_score = 0.0;
	for i in 0..num_tests
	{
		let mut sample = isolation_forest::Sample::new();
		let mut features = isolation_forest::FeatureList::new();

		let x = range1.sample(&mut rng) as u64;
		let y = range1.sample(&mut rng) as u64;

		features.push(isolation_forest::Feature::new("x", x));
		features.push(isolation_forest::Feature::new("y", y));
		sample.add_features(&mut features);

		// Run a test with the sample that doesn't contain outliers.
		let score = forest.score(&sample);
		avg_control_score = avg_control_score + score;
        let normalized_score = forest.normalized_score(&sample);
        avg_control_normalized_score = avg_control_normalized_score + normalized_score;
		println!("Control test sample {}: {:.2} {:.2} {:.2} {:.2}", i, x, y, score, normalized_score);
	}
	avg_control_score = avg_control_score / num_tests as f64;
	avg_control_normalized_score = avg_control_normalized_score / num_tests as f64;
    println!("Average of control test samples: {:.2}.", avg_control_score);
    println!("Average of control test samples (normalized): {:.2}.", avg_control_normalized_score);

	// Outlier samples (different from training samples).
	println!("\nTest samples that are different from the training set.");
	println!("------------------------------------------------------");
	let mut avg_outlier_score = 0.0;
	let mut avg_outlier_normalized_score = 0.0;
	for i in 0..num_tests
	{
		let mut sample = isolation_forest::Sample::new();
		let mut features = isolation_forest::FeatureList::new();

		let x = range2.sample(&mut rng) as u64;
		let y = range2.sample(&mut rng) as u64;

		features.push(isolation_forest::Feature::new("x", x));
		features.push(isolation_forest::Feature::new("y", y));
		sample.add_features(&mut features);

		// Run a test with the sample that contains outliers.
		let score = forest.score(&sample);
		avg_outlier_score = avg_outlier_score + score;
        let normalized_score = forest.normalized_score(&sample);
        avg_outlier_normalized_score = avg_outlier_normalized_score + normalized_score;
		println!("Outlier test sample {}: {:.2} {:.2} {:.2} {:.2}", i, x, y, score, normalized_score);
	}
	avg_outlier_score = avg_outlier_score / num_tests as f64;
	avg_outlier_normalized_score = avg_outlier_normalized_score / num_tests as f64;
    println!("Average of outlier test samples: {:.2}.", avg_outlier_score);
    println!("Average of outlier test samples (normalized): {:.2}.", avg_outlier_normalized_score);

    if dump == true {
        println!("{}", forest.dump());
    }
}
