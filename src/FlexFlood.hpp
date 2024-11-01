#pragma once
#include <vector>
#include "./RandomNumberGenerator.hpp"
#include "./cpp-btree/btree_set.h"
template <typename Type, int dimension>
struct FlexFlood {
    using T = std::array<Type, dimension>;
    using U = std::array<int, dimension>;
    using Iter = typename std::vector<T>::iterator;
    FlexFlood(const Iter& begin, const Iter& end, const std::vector<T>& sample_lowers, const std::vector<T>& sample_uppers, const int num_sampling = 30 * dimension, const double learning_time = 60.0 * dimension)
        : data(0), num_data(end - begin), divideSize(), divideSizeSigma(0), grid(dimension), count_data(dimension) {
        assert(sample_lowers.size() == sample_uppers.size());
        assert(learning_time >= 0.0);
        learn(begin, end, sample_lowers, sample_uppers, num_sampling, learning_time);
    }
    FlexFlood(const Iter& begin, const Iter& end, const U& partitioning)
        : data(0), num_data(end - begin), divideSize(partitioning), divideSizeSigma(1), grid(dimension), count_data(dimension) {
        int sort_dimension = -1;
        for(int d = 0; d < dimension; ++d) {
            assert(partitioning[d] > 0);
            if(partitioning[d] == 1) {
                sort_dimension = d;
            }
        }
        assert(sort_dimension != -1);
        init(begin, end, divideSize, sort_dimension);
    }
    inline void clear() {
        num_data = 0;
        for(int i = 0; i < divideSizeSigma; ++i) {
            delete data[i];
        }
        data.resize(0);
        divideSize = {};
        divideSize_ini = {};
        divideSizeSigma = 0;
        for(int i = 0; i < dimension; ++i) grid[i].clear();
        for(int i = 0; i < dimension; ++i) count_data[i].clear();
        sortDimension = -1;
    }
    inline size_t size() {
        return num_data;
    }
    std::vector<T> enumerate(const T& lower, const T& upper) {
        std::vector<T> res;
        std::vector<int> idx = projection(lower, upper);
        for(const int i : idx) {
            std::pair<typename btree::btree_set<std::pair<double, T>>::iterator, typename btree::btree_set<std::pair<double, T>>::iterator> seg = refinement(lower, upper, i);
            for(auto iter = seg.first; iter != seg.second; ++iter) {
                scan(lower, upper, iter, res);
            }
        }
        return res;
    }
    void insert(const T& dat, const std::pair<int, int>& lower = std::make_pair(3, 1), const std::pair<int, int>& upper = std::make_pair(1, 2)) {
        const std::pair<int, int> middle = {lower.first * upper.first, lower.second * upper.first + upper.second * lower.first};
        std::vector<int> idx(dimension);
        int index = 0, sigma = 1;
        for(int i = dimension - 1; i >= 0; --i) {
            if(i == sortDimension) continue;
            idx[i] = upper_bound(grid[i].begin(), grid[i].end(), dat[i]) - grid[i].begin() - 1;
            index += idx[i] * sigma;
            sigma *= divideSize[i];
        }
        int prev_size = data[index]->size();
        data[index]->insert(make_pair(dat[sortDimension], dat));
        if(prev_size == data[index]->size()) return;
        ++num_data;
        for(int i = 0; i < dimension; ++i) {
            ++count_data[i][idx[i]];
            if(lower.first * count_data[i][idx[i]] * divideSize_ini[i] <= lower.second * num_data) {
                if(idx[i] == 0) {
                    if(middle.first * count_data[i][idx[i] + 1] * divideSize_ini[i] <= middle.second * num_data) {
                        merge(i, idx[i]);
                    } else {
                        equalize(i, idx[i]);
                    }
                } else {
                    if(middle.first * count_data[i][idx[i] - 1] * divideSize_ini[i] <= middle.second * num_data) {
                        merge(i, idx[i] - 1);
                    } else {
                        equalize(i, idx[i] - 1);
                    }
                }
            } else if(upper.first * count_data[i][idx[i]] * divideSize_ini[i] >= upper.second * num_data) {
                split(i, idx[i]);
            }
        }
    }
    void erase(const T& dat, const std::pair<int, int>& lower = std::make_pair(3, 1), const std::pair<int, int>& upper = std::make_pair(1, 2)) {
        const std::pair<int, int> middle = {lower.first * upper.first, lower.second * upper.first + upper.second * lower.first};
        std::vector<int> idx(dimension);
        int index = 0, sigma = 1;
        for(int i = dimension - 1; i >= 0; --i) {
            if(i == sortDimension) continue;
            idx[i] = upper_bound(grid[i].begin(), grid[i].end(), dat[i]) - grid[i].begin() - 1;
            index += idx[i] * sigma;
            sigma *= divideSize[i];
        }
        int prev_size = data[index]->size();
        data[index]->erase(make_pair(dat[sortDimension], dat));
        if(prev_size == data[index]->size()) return;
        --num_data;
        for(int i = 0; i < dimension; ++i) {
            --count_data[i][idx[i]];
            if(lower.first * count_data[i][idx[i]] * divideSize_ini[i] <= lower.second * num_data) {
                if(idx[i] == 0) {
                    if(middle.first * count_data[i][idx[i] + 1] * divideSize_ini[i] <= middle.second * num_data) {
                        merge(i, idx[i]);
                    } else {
                        equalize(i, idx[i]);
                    }
                } else {
                    if(middle.first * count_data[i][idx[i] - 1] * divideSize_ini[i] <= middle.second * num_data) {
                        merge(i, idx[i] - 1);
                    } else {
                        equalize(i, idx[i] - 1);
                    }
                }
            } else if(upper.first * count_data[i][idx[i]] * divideSize_ini[i] >= upper.second * num_data) {
                split(i, idx[i]);
            }
        }
    }

   private:
    static constexpr double eps = 1e-9;
    std::vector<btree::btree_set<std::pair<double, T>>*> data;
    int num_data;
    U divideSize, divideSize_ini;
    int divideSizeSigma;
    std::vector<std::vector<double>> grid;
    std::vector<std::vector<int>> count_data;
    int sortDimension;
    void learn(const Iter& begin, const Iter& end, const std::vector<T>& sample_lower, const std::vector<T>& sample_upper, const double learning_time, const int num_sampling) {
        const std::filesystem::path file_path = __FILE__;
        const std::string path = file_path.parent_path().string();
        const std::string sample_path = path + "/../train/sample.csv";
        const std::string train_path = path + "/../train/train.csv";
        const std::string command = "python3 " + path + "/learning.py " + std::to_string(learning_time) + ' ' + sample_path + " > " + train_path;
        std::ofstream sampleFile(sample_path);
        std::ifstream trainFile(train_path);
        for(int sortdim = 0; sortdim < dimension; ++sortdim) {
            for(int i = 0; i < num_sampling / dimension + int(sortdim < (num_sampling % dimension)); ++i) {
                std::array<int, dimension> sample_partitioning;
                for(int d = 0; d < dimension; ++d) {
                    sample_partitioning[d] = rng(2, 100);
                }
                sample_partitioning[sortdim] = 1;
                init(begin, end, sample_partitioning, sortdim);
                const std::tuple<int, int, int> sample_time = measure_time(sample_lower, sample_upper, sampleFile);
                for(int d = 0; d < dimension; ++d) {
                    sampleFile << sample_partitioning[d] << ' ';
                }
                sampleFile << std::get<0>(sample_time) << ' ' << std::get<1>(sample_time) << ' ' << std::get<2>(sample_time) << '\n';
                clear();
            }
        }
        sampleFile.close();
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
        if(!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        std::array<int, dimension> partitioning;
        int sort_dim = -1;
        for(int d = 0; d < dimension; ++d) {
            trainFile >> partitioning[d];
            if(partitioning[d] == 1) {
                sort_dim = d;
            }
        }
        trainFile.close();
        init(begin, end, partitioning, sort_dim);
    }
    std::tuple<int, int, int> measure_time(const std::vector<T>& lowers, const std::vector<T>& uppers, std::ofstream& sampleFile) {
        const int num_query = (int)lowers.size();
        int w_p = 0, w_r = 0, w_s = 0;
        int sum = 0;
        std::chrono::system_clock::time_point start, end;
        std::chrono::nanoseconds time;
        std::chrono::_V2::steady_clock::rep msec;
        for(int q = 0; q < num_query; ++q) {
            std::vector<T> res;
            start = std::chrono::system_clock::now();
            std::vector<int> idx = projection(lowers[q], uppers[q]);
            end = std::chrono::system_clock::now();
            time = end - start;
            msec = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
            w_p += msec;
            std::vector<std::pair<typename btree::btree_set<std::pair<double, T>>::iterator, typename btree::btree_set<std::pair<double, T>>::iterator>> seg(idx.size());
            start = std::chrono::system_clock::now();
            for(int i = 0; i < (int)idx.size(); ++i) {
                seg[i] = refinement(lowers[q], uppers[q], idx[i]);
            }
            end = std::chrono::system_clock::now();
            time = end - start;
            msec = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
            w_r += msec;
            start = std::chrono::system_clock::now();
            for(int i = 0; i < (int)seg.size(); ++i) {
                for(auto iter = seg[i].first; iter != seg[i].second; ++iter) {
                    scan(lowers[q], uppers[q], iter, res);
                }
            }
            end = std::chrono::system_clock::now();
            time = end - start;
            msec = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
            w_s += msec;
            sum += res.size();
        }
        sampleFile << sum << '\n';
        return std::make_tuple(w_p, w_r, w_s);
    }
    void init(const Iter& begin, const Iter& end, const U& divsize, const int sortdim) {
        num_data = int(end - begin);
        sortDimension = sortdim;
        divideSizeSigma = 1;
        for(int i = 0; i < dimension; ++i) {
            divideSize[i] = divsize[i];
            divideSize_ini[i] = divideSize[i];
            divideSizeSigma *= divsize[i];
            grid[i].resize(divsize[i] + 1);
            count_data[i].resize(divsize[i]);
        }
        data.resize(divideSizeSigma);
        for(int i = 0; i < divideSizeSigma; ++i) {
            data[i] = new btree::btree_set<std::pair<double, T>>;
        }
        for(int i = 0; i < dimension; ++i) {
            std::vector<double> vec(num_data);
            int index = 0;
            for(auto iter = begin; iter != end; ++iter) {
                vec[index] = (*iter)[i];
                ++index;
            }
            std::sort(vec.begin(), vec.end());
            grid[i][0] = -1e100;
            for(int j = 1; j < divsize[i]; ++j) {
                grid[i][j] = vec[num_data * j / divsize[i] - 1] + eps;
            }
            grid[i][divsize[i]] = 1e100;
        }
        for(auto iter = begin; iter != end; ++iter) {
            int index = 0;
            int remDim = divideSizeSigma;
            for(int i = 0; i < dimension; ++i) {
                if(i == sortDimension) {
                    ++count_data[i][0];
                    continue;
                }
                remDim /= divsize[i];
                int idx = upper_bound(grid[i].begin(), grid[i].end(), (*iter)[i]) - grid[i].begin();
                --idx;
                index += idx * remDim;
                ++count_data[i][idx];
            }
            data[index]->insert(std::make_pair((*iter)[sortDimension], *iter));
        }
    }
    inline std::vector<int> projection(const T& lower, const T& upper) {
        U left, right, cur, plus;
        int sz = 1, idx = 0, sigma = 1;
        for(int d = dimension - 1; d >= 0; --d) {
            left[d] = upper_bound(grid[d].begin(), grid[d].end(), lower[d]) - grid[d].begin() - 1;
            right[d] = lower_bound(grid[d].begin(), grid[d].end(), upper[d]) - grid[d].begin();
            plus[d] = sigma;
            cur[d] = left[d];
            idx += left[d] * plus[d];
            sz *= (right[d] - left[d]);
            sigma *= divideSize[d];
        }
        std::vector<int> res(sz);
        res[0] = idx;
        for(int i = 1; i < sz; ++i) {
            ++cur[dimension - 1];
            ++idx;
            for(int d = dimension - 1; cur[d] == right[d]; --d) {
                cur[d] = left[d];
                idx -= (right[d] - left[d]) * plus[d];
                ++cur[d - 1];
                idx += plus[d - 1];
            }
            res[i] = idx;
        }
        return res;
    }
    inline typename std::pair<typename btree::btree_set<std::pair<double, T>>::iterator, typename btree::btree_set<std::pair<double, T>>::iterator> refinement(const T& lower, const T& upper, const int& i) {
        auto l = data[i]->lower_bound(make_pair(lower[sortDimension] - eps, T{})), r = data[i]->lower_bound(make_pair(upper[sortDimension] + eps, T{}));
        return make_pair(l, r);
    }
    inline void scan(const T& lower, const T& upper, const typename btree::btree_set<std::pair<double, T>>::const_iterator& iter, std::vector<T>& res) {
        for(int j = 0; j < dimension; ++j) {
            if(!(lower[j] <= (*iter).second[j] and (*iter).second[j] <= upper[j])) {
                return;
            }
        }
        res.push_back((*iter).second);
    }
    void insert_only(const T& dat) {
        int index = 0, sigma = 1;
        for(int i = dimension - 1; i >= 0; --i) {
            if(i == sortDimension) continue;
            int idx = upper_bound(grid[i].begin(), grid[i].end(), dat[i]) - grid[i].begin() - 1;
            index += idx * sigma;
            sigma *= divideSize[i];
        }
        data[index]->insert(make_pair(dat[sortDimension], dat));
    }
    void split(const int dim, const int idx) {
        std::vector<T> dat;
        U left = {}, right = divideSize, cur = {}, plus;
        left[dim] = idx, right[dim] = idx + 1, cur[dim] = idx;
        int sz = 1, index = 0, sigma = 1;
        for(int d = dimension - 1; d >= 0; --d) {
            plus[d] = sigma;
            index += left[d] * plus[d];
            sz *= (right[d] - left[d]);
            sigma *= divideSize[d];
        }
        for(const auto& x : *data[index]) {
            dat.push_back(x.second);
        }
        data[index]->clear();
        for(int i = 1; i < sz; ++i) {
            ++cur[dimension - 1];
            ++index;
            for(int d = dimension - 1; cur[d] == right[d]; --d) {
                cur[d] = left[d];
                index -= (right[d] - left[d]) * plus[d];
                ++cur[d - 1];
                index += plus[d - 1];
            }
            for(const auto& x : *data[index]) {
                dat.push_back(x.second);
            }
            data[index]->clear();
        }
        sort(dat.begin(), dat.end(), [&](const T& x, const T& y) { return x[dim] < y[dim]; });
        divideSizeSigma /= divideSize[dim];
        ++divideSize[dim];
        divideSizeSigma *= divideSize[dim];
        count_data[dim][idx] = (int)dat.size() / 2;
        count_data[dim].insert(count_data[dim].begin() + idx + 1, ((int)dat.size() + 1) / 2);
        grid[dim].insert(grid[dim].begin() + idx + 1, dat[(int)dat.size() / 2][dim] - eps);
        std::vector<btree::btree_set<std::pair<double, T>>*> new_data(divideSizeSigma);
        left = {}, right = divideSize, cur = {};
        int cnt = 0;
        for(int i = 0; i < divideSizeSigma; ++i) {
            if(cur[dim] == idx + 1) {
                new_data[i] = new btree::btree_set<std::pair<double, T>>;
                ++cnt;
            } else {
                new_data[i] = data[i - cnt];
            }
            ++cur[dimension - 1];
            for(int d = dimension - 1; d >= 1 and cur[d] == right[d]; --d) {
                cur[d] = left[d];
                ++cur[d - 1];
            }
        }
        swap(data, new_data);
        for(const T& x : dat) {
            insert_only(x);
        }
    }
    void merge(const int dim, const int idx) {
        std::vector<T> dat;
        U left = {}, right = divideSize, cur = {}, plus;
        left[dim] = idx, right[dim] = idx + 1, cur[dim] = idx;
        int sz = 1, index = 0, sigma = 1;
        for(int d = dimension - 1; d >= 0; --d) {
            plus[d] = sigma;
            index += left[d] * plus[d];
            sz *= (right[d] - left[d]);
            sigma *= divideSize[d];
        }
        for(const auto& x : *data[index]) {
            dat.push_back(x.second);
        }
        delete data[index];
        for(int i = 1; i < sz; ++i) {
            ++cur[dimension - 1];
            ++index;
            for(int d = dimension - 1; cur[d] == right[d]; --d) {
                cur[d] = left[d];
                index -= (right[d] - left[d]) * plus[d];
                ++cur[d - 1];
                index += plus[d - 1];
            }
            for(const auto& x : *data[index]) {
                dat.push_back(x.second);
            }
            delete data[index];
        }
        left = {}, right = divideSize, cur = {};
        int old_sigma = divideSizeSigma;
        divideSizeSigma /= divideSize[dim];
        --divideSize[dim];
        divideSizeSigma *= divideSize[dim];
        count_data[dim].erase(count_data[dim].begin() + idx);
        count_data[dim][idx] += (int)dat.size();
        grid[dim].erase(grid[dim].begin() + idx + 1);
        std::vector<btree::btree_set<std::pair<double, T>>*> new_data(divideSizeSigma);
        int cnt = 0;
        for(int i = 0; i < old_sigma; ++i) {
            if(cur[dim] == idx) {
                ++cnt;
            } else {
                new_data[i - cnt] = data[i];
            }
            ++cur[dimension - 1];
            for(int d = dimension - 1; d >= 1 and cur[d] == right[d]; --d) {
                cur[d] = left[d];
                ++cur[d - 1];
            }
        }
        swap(data, new_data);
        for(const T& x : dat) {
            insert_only(x);
        }
    }
    void equalize(const int dim, const int idx) {
        std::vector<T> dat;
        U left = {}, right = divideSize, cur = {}, plus;
        left[dim] = idx, right[dim] = idx + 2, cur[dim] = idx;
        int sz = 1, index = 0, sigma = 1;
        for(int d = dimension - 1; d >= 0; --d) {
            plus[d] = sigma;
            index += left[d] * plus[d];
            sz *= (right[d] - left[d]);
            sigma *= divideSize[d];
        }
        for(const auto& x : *data[index]) {
            dat.push_back(x.second);
        }
        data[index]->clear();
        for(int i = 1; i < sz; ++i) {
            ++cur[dimension - 1];
            ++index;
            for(int d = dimension - 1; cur[d] == right[d]; --d) {
                cur[d] = left[d];
                index -= (right[d] - left[d]) * plus[d];
                ++cur[d - 1];
                index += plus[d - 1];
            }
            for(const auto& x : *data[index]) {
                dat.push_back(x.second);
            }
            data[index]->clear();
        }
        std::sort(dat.begin(), dat.end(), [&](const T& x, const T& y) { return x[dim] < y[dim]; });
        count_data[dim][idx] = (int)dat.size() / 2;
        count_data[dim][idx + 1] = ((int)dat.size() + 1) / 2;
        grid[dim][idx + 1] = dat[(int)dat.size() / 2][dim] - eps;
        for(const T& x : dat) {
            insert_only(x);
        }
    }
};