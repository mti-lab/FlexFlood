# FlexFlood

[日本語版README](https://github.com/mti-lab/FlexFlood/blob/main/README_jp.md)

This is a multi-dimensional learned index that processes the following queries efficiently.

- Enumerate data points inside a hyperrectangle parallel to the axes
- Add/Delete data points

We used `gnu++17` and `python3` for the internal implementation.<br>
We also used `scikit-learn`.

The sample code is `main.cpp`.<br>
First, generate a sample dataset using the following command.

```
./data_generator.sh 3 100000 1000 2000000 ./data/data.csv ./data/sample_query.csv ./data/query.csv
```

After that, `main.cpp` can be executed by the following command.

```
./running.sh main.cpp ./data/data.csv ./data/sample_query.csv ./data/query.csv
```

The following is a detailed document.<br>
Sections marked with 💻 are for those who understand the internal algorithm.<br>
We designed it so that even if the sections marked with 💻 were ignored, it would not be a problem to use it as a black box.

## constructor

```cpp
(1) FlexFlood<typename Type, int dimension> index(std::vector<std::array<Type, dimension>>::iterator begin, std::vector<std::array<Type, dimension>>::iterator end, std::vector<std::array<Type, dimension>> sample_lowers, std::vector<std::array<Type, dimension>> sample_uppers, int num_sampling = 30 * dimension, double learning_time = 60.0 * dimension)
(2 💻) FlexFlood<typename Type, int dimension> index(std::vector<std::array<Type, dimension>>::iterator begin, std::vector<std::array<Type, dimension>>::iterator end, std::array<int, dimension> partitioning)
```

- Constructs a FlexFlood `index` from `dimension`-dimensional data of type `Type` between `begin` and `end`.
- (1) Optimizes the internal structure based on sample queries. Provide the coordinates of the lower corners of the hyperrectangles in `sample_lowers` and the upper corners in `sample_uppers`.
- (1) 💻 The number of samples for learning the distribution can be specified with `num_sampling`, and the learning time with `learning_time`.
- (2) Initializes based on the number of partitions for each axis specified in `partitioning`. Note that at least one value in `partitioning` must be $1$ for the internal algorithm to function properly.

**Constraints**

- (1) `sample_lowers` and `sample_uppers` must have the same length.
- (2) At least one value in `partitioning` must be $1$.
- (2) Each value in `partitioning` must be positive.

## clear

```cpp
void index.clear()
```

Deletes all elements in `index`.

## size

```cpp
size_t index.size()
```

Returns the number of elements in `index`.

## enumerate

```cpp
std::vector<std::array<Type, dimension>> enumerate(std::array<Type, dimension> lower, std::array<Type, dimension> upper)
```

Returns all data points within `index` that are inside the hyperrectangle defined by the `lower` and `upper` corners.

**Constraints**

- For each dimension, the value of `lower` must be less than or equal to the value of `upper` (behavior is undefined if this condition is not met).


## insert

```cpp
void insert(std::array<Type, dimension> dat, std::pair<int, int> lower = std::make_pair(3, 1), std::pair<int, int> upper = std::make_pair(1, 2))
```

- Adds `dat` to `index`.
- If `index` already contains `dat`, no action is taken, and `size()` remains unchanged.
- 💻 `lower` and `upper` specify thresholds for internal processing of `split` and `merge/equalize`.
- 💻 `lower = std::make_pair(a, b)` means that the threshold coefficient for `split` is $\frac{b}{a}$.
- 💻 `upper = std::make_pair(a, b)` means that the threshold coefficient for `merge/equalize` is $\frac{b}{a}$.

## erase

```cpp
void erase(std::array<Type, dimension> dat, std::pair<int, int> lower = std::make_pair(3, 1), std::pair<int, int> upper = std::make_pair(1, 2))
```

- Deletes `dat` from `index`.
- If `index` does not contain `dat`, no action is taken, and `size()` remains unchanged.
- 💻 `lower` and `upper` specify thresholds for internal processing of `split` and `merge/equalize`.
- 💻 `lower = std::make_pair(a, b)` means that the threshold coefficient for `split` is $\frac{b}{a}$.
- 💻 `upper = std::make_pair(a, b)` means that the threshold coefficient for `merge/equalize` is $\frac{b}{a}$.
