[![C++](https://img.shields.io/badge/cpp-brightgreen.svg)]() [![Rust](https://img.shields.io/badge/rust-brightgreen.svg)](https://www.rust-lang.org) [![Python 2.7|3.7](https://img.shields.io/badge/python-2.7%2F3.7-brightgreen.svg)](https://www.python.org/) [![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

# LibIsolationForest

## Description

This project contains Rust, C++, and python implementations of the Isolation Forest algorithm. Isolation Forest is an anomaly detection algorithm based around a collection of randomly generated decision trees. For a full description of the algorithm, consult the original paper by the algorithm's creators:

https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

## Python Example

The python implementation can be installed via pip:

```pip install IsolationForest```

Here's a short code snipet that shows how to use the Python version of the library. You can also read the file `test.py` for a complete example. As the library matures, I'll add more test examples to this file.

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

An example of how to use the Rust version of the library can be found in `main.rs`. As the library matures, I'll add more test examples to this file.

## C++ Example

An example of how to use the C++ version of the library can be found in `main.cpp`. As the library matures, I'll add more test examples to this file.

## License

This library is released under the MIT license, see LICENSE for details.
