# LibIsolationForest

## Description

This project contains C++ and python implementations of the Isolation Forest algorithm. Isolation Forest is an anomaly detection algorithm based around a collection of randomly generated decision trees. For a full description of the algorithm, consult the original paper by the algorithm's creators:

https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

## Python Example

Here's a short code snipet that shows how to use the Python version of the library. You can also read the file `test.py` for a complete example. As the library matures, I'll add more test examples to this file.

```python
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
```
