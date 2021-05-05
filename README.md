[![C++](https://img.shields.io/badge/cpp-brightgreen.svg)]() [![Rust](https://img.shields.io/badge/rust-brightgreen.svg)](https://www.rust-lang.org) [![Python 2.7|3.7](https://img.shields.io/badge/python-2.7%2F3.7-brightgreen.svg)](https://www.python.org/) [![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)![](https://travis-ci.com/msimms/LibIsolationForest.svg?branch=master)

# LibIsolationForest

## Description

This project contains Rust, C++, Julia, and python implementations of the Isolation Forest algorithm. Isolation Forest is an anomaly detection algorithm based around a collection of randomly generated decision trees. For a full description of the algorithm, consult the original paper by the algorithm's creators:

https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

## Python Example

The python implementation can be installed via pip:

```pip install IsolationForest```

This is a short code snipet that shows how to use the Python version of the library. You can also read the file `test.py` for a complete example. As the library matures, I'll add more test examples to this file.

```python
from isolationforest import IsolationForest

forest = IsolationForest.Forest(num_trees, sub_sampling_size)

sample = IsolationForest.Sample("Training Sample 1")
features = []
features.append({"feature 1": feature_1_value})
# Add more features to the sample...
features.append({"feature N": feature_N_value})
sample.add_features(features)
# Add the features to the sample.
forest.add_sample(sample)
# Add more samples to the forest...

# Create the forest.
forest.create()

sample = IsolationForest.Sample("Test Sample 1")
features = []
features.append({"feature 1": feature_1_value})
# Add more features to the sample...
features.append({"feature N": feature_N_value})
# Add the features to the sample.
sample.add_features(features)

# Score the sample.
score = forest.score(sample)
normalized_score = forest.normalized_score(sample)
```

## Rust Example

Add `isolation_forest` to your `Cargo.toml` file.

More examples of how to use the Rust version of the library can be found in `lib.rs`. As the library matures, I'll add more test examples to this file.

```rust
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
```

## C++ Example

An example of how to use the C++ version of the library can be found in `main.cpp`. As the library matures, I'll add more test examples to this file.

## Julia Example

An example of how to use the Julia version of the library can be found in `test.jl`. As the library matures, I'll add more test examples to this file.

## Version History

### 1.0
* Initial version.

### 1.1
* Added normalized scores.
* Updated random number generation in rust, because it changed again.

## License

This library is released under the MIT license, see LICENSE for details.
