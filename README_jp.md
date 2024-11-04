# FlexFlood

[English version of README](https://github.com/mti-lab/FlexFlood/blob/main/README.md)

- 軸に平行な超直方体内部のデータ点の列挙
- データ点の追加・削除

を高速に処理する学習型多次元索引です。

内部実装は `gnu++17` および `python3` です。<br>
また `scikit-learn` を用いています。

サンプルコードを `main.cpp` に示します。<br>
実行する場合、はじめに `./data_generator.sh 3 100000 1000 2000000 ./data/data.csv ./data/sample_query.csv ./data/query.csv` によりサンプルデータセットを生成してください。<br>
その後 `./running.sh main.cpp ./data/data.csv ./data/sample_query.csv ./data/query.csv` によって実行できます。

以下は詳細なドキュメントです。<br>
💻 がついている項は、内部アルゴリズムを理解している方向けの説明です。<br>
このマークがついた項を無視しても、ブラックボックスとして使用する分には困らないように設計されています。

## コンストラクタ

```cpp
(1) FlexFlood<typename Type, int dimension> index(std::vector<std::array<Type, dimension>>::iterator begin, std::vector<std::array<Type, dimension>>::iterator end, std::vector<std::array<Type, dimension>> sample_lowers, std::vector<std::array<Type, dimension>> sample_uppers, int num_sampling = 30 * dimension, double learning_time = 60.0 * dimension)
(2 💻) FlexFlood<typename Type, int dimension> index(std::vector<std::array<Type, dimension>>::iterator begin, std::vector<std::array<Type, dimension>>::iterator end, std::array<int, dimension> partitioning)
```

- `begin` と `end` の間にある `Type` 型の `dimension` 次元データをもとにFlexFlood `index` を構築します。
- (1) サンプルクエリに基づいて内部構造を最適化します。サンプルクエリの超直方体の最小点の座標を `sample_lowers` に、最大点の座標を `sample_uppers` に与えてください。
- (1) 💻 分布学習のための内部構造のサンプリング回数を `num_sampling` で、学習にかける時間を `learning_time` で指定できます。
- (2) 各軸の分割数を `partitioning` に従って初期化します。内部アルゴリズムの都合上 `partitioning` の値の少なくとも1つは $1$ でないといけないことに注意してください。

**制約**

- (1) `sample_lowers` と `sample_uppers` の長さは等しい。
- (2) `partitioning` の値の少なくとも1つは $1$ である。
- (2) `partitioning` の各値は正である。

## clear

```cpp
void index.clear()
```

`index` の全ての要素を削除します。

## size

```cpp
size_t index.size()
```

`index` 内の要素数を返します。

## enumerate

```cpp
std::vector<std::array<Type, dimension>> enumerate(std::array<Type, dimension> lower, std::array<Type, dimension> upper)
```

最小点・最大点がそれぞれ `lower` と `upper` であるような超直方体領域内部に存在する `index` 内のデータ点をすべて返します。

**制約**

- 各次元について `lower` の値は `upper` の値以下である (成立しない場合の挙動は未定義です)。

## insert

```cpp
void insert(std::array<Type, dimension> dat, std::pair<int, int> lower = std::make_pair(3, 1), std::pair<int, int> upper = std::make_pair(1, 2))
```

- `index` にデータ点 `dat` を追加します。
- `index` に `dat` を全く同じデータ点が含まれている場合は何も行いません。この場合は `size()` も不変です。
- 💻 `lower` と `upper` はそれぞれ内部処理の ``split`` と ``merge/equalize`` を実行するための閾値です。
- 💻 `lower = std::make_pair(a, b)` のとき `split` の閾値の係数が $\frac{b}{a}$ であることを意味します。
- 💻 `upper = std::make_pair(a, b)` のとき `merge/euqalize` の閾値の係数が $\frac{b}{a}$ であることを意味します。

## erase

```cpp
void erase(std::array<Type, dimension> dat, std::pair<int, int> lower = std::make_pair(3, 1), std::pair<int, int> upper = std::make_pair(1, 2))
```

- `index` からデータ点 `dat` を削除します。
- `index` に `dat` が含まれない場合は何も行いません。この場合は `size()` も不変です。
- 💻 `lower` と `upper` はそれぞれ内部処理の ``split`` と ``merge/equalize`` を実行するための閾値です。
- 💻 `lower = std::make_pair(a, b)` のとき `split` の閾値の係数が $\frac{b}{a}$ であることを意味します。
- 💻 `upper = std::make_pair(a, b)` のとき `merge/euqalize` の閾値の係数が $\frac{b}{a}$ であることを意味します。
