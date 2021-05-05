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

pub mod isolation_forest;

#[cfg(test)]
mod tests {
    extern crate csv;
    extern crate rand;

    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn random_test() {
        let args: Vec<String> = std::env::args().collect();
        let num_tests = 10;
        let num_training_samples = 10;
        let mut forest = crate::isolation_forest::Forest::new(10, 10);
        let mut rng = rand::thread_rng();
        let range1 = Uniform::from(0..25);
        let range2 = Uniform::from(15..45);
        let mut dump = false;

        for arg in args {
            if arg == "--dump" {
                dump = true;
            }
        }

        // Training samples.
        for _i in 0..num_training_samples {
            let x = range1.sample(&mut rng) as u64;
            let y = range1.sample(&mut rng) as u64;

            let mut features = crate::isolation_forest::FeatureList::new();
            features.push(crate::isolation_forest::Feature::new("x", x));
            features.push(crate::isolation_forest::Feature::new("y", y));

            let mut sample = crate::isolation_forest::Sample::new("training");
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
        for i in 0..num_tests {
            let x = range1.sample(&mut rng) as u64;
            let y = range1.sample(&mut rng) as u64;

            let mut features = crate::isolation_forest::FeatureList::new();
            features.push(crate::isolation_forest::Feature::new("x", x));
            features.push(crate::isolation_forest::Feature::new("y", y));

            let mut sample = crate::isolation_forest::Sample::new("normal");
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
        for i in 0..num_tests {

            let x = range2.sample(&mut rng) as u64;
            let y = range2.sample(&mut rng) as u64;

            let mut features = crate::isolation_forest::FeatureList::new();
            features.push(crate::isolation_forest::Feature::new("x", x));
            features.push(crate::isolation_forest::Feature::new("y", y));

            let mut sample = crate::isolation_forest::Sample::new("outlier");
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

        assert!(avg_control_normalized_score < avg_outlier_normalized_score);
    }

    #[test]
    fn iris_test() {
        let file_path = "../data/iris.data.txt";
        let file = match std::fs::File::open(&file_path) {
            Err(why) => panic!("Couldn't open {} {}", file_path, why),
            Ok(file) => file,
        };

        let mut reader = csv::Reader::from_reader(file);
        let mut forest = crate::isolation_forest::Forest::new(10, 10);
        let training_class_name = "Iris-setosa";
        let mut training_samples = Vec::new();
        let mut test_samples = Vec::new();
        let mut avg_control_set_score = 0.0;
        let mut avg_outlier_set_score = 0.0;
        let mut avg_control_set_normalized_score = 0.0;
        let mut avg_outlier_set_normalized_score = 0.0;
        let mut num_control_tests = 0;
        let mut num_outlier_tests = 0;
        let mut rng = rand::thread_rng();
        let range = Uniform::from(0..10);

        for record in reader.records() {
            let record = record.unwrap();

            let sepal_length_cm: f64 = record[0].parse().unwrap();
            let sepal_width_cm: f64 = record[1].parse().unwrap();
            let petal_length_cm: f64 = record[2].parse().unwrap();
            let petal_width_cm: f64 = record[3].parse().unwrap();
            let name: String = record[4].parse().unwrap();

            let mut features = crate::isolation_forest::FeatureList::new();
            features.push(crate::isolation_forest::Feature::new("sepal length in cm", (sepal_length_cm * 10.0) as u64));
            features.push(crate::isolation_forest::Feature::new("sepal width in cm", (sepal_width_cm * 10.0) as u64));
            features.push(crate::isolation_forest::Feature::new("petal length in cm", (petal_length_cm * 10.0) as u64));
            features.push(crate::isolation_forest::Feature::new("petal width in cm", (petal_width_cm * 10.0) as u64));

            let mut sample = crate::isolation_forest::Sample::new(&name);
            sample.add_features(&mut features);

            // Randomly split the samples into training and test samples.
            let x = range.sample(&mut rng) as u64;
            if x > 5 && name == training_class_name {
                forest.add_sample(sample.clone());
                training_samples.push(sample);
            }
            else {
                test_samples.push(sample);
            }
        }

        // Create the forest.
        forest.create();

        // Use each test sample.
        for test_sample in test_samples {
            let score = forest.score(&test_sample);
            let normalized_score = forest.normalized_score(&test_sample);

            if training_class_name == test_sample.name {
                avg_control_set_score = avg_control_set_score + score;
                avg_control_set_normalized_score = avg_control_set_normalized_score + normalized_score;
                num_control_tests = num_control_tests + 1;
            }
            else {
                avg_outlier_set_score = avg_outlier_set_score + score;
                avg_outlier_set_normalized_score = avg_outlier_set_normalized_score + normalized_score;
                num_outlier_tests = num_outlier_tests + 1;
            }
        }

        // Compute statistics.
        if num_control_tests > 0 {
            avg_control_set_score = avg_control_set_score / num_control_tests as f64;
            avg_control_set_normalized_score = avg_control_set_normalized_score / num_control_tests as f64;
        }
        if num_outlier_tests > 0 {
            avg_outlier_set_score = avg_outlier_set_score / num_outlier_tests as f64;
            avg_outlier_set_normalized_score = avg_outlier_set_normalized_score / num_outlier_tests as f64;
        }

        println!("Avg Control Score: {}", avg_control_set_score);
        println!("Avg Control Normalized Score: {}", avg_control_set_normalized_score);
        println!("Avg Outlier Score: {}", avg_outlier_set_score);
        println!("Avg Outlier Normalized Score: {}", avg_outlier_set_normalized_score);

        assert!(avg_control_set_normalized_score < avg_outlier_set_normalized_score);
    }
}
