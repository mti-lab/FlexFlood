#pragma once
#include <bits/stdc++.h>
#include "RandomForestRegression.hpp"
#include "./cpp-btree/btree_set.h"
struct RealRandomNumberGenerator {
    std::mt19937_64 mt;
    RealRandomNumberGenerator()
        : mt(std::chrono::steady_clock::now().time_since_epoch().count()) {}
    inline double operator()(const double l, const double r) {
        std::uniform_real_distribution<double> dist(l, r);
        return dist(mt);
    }
} rrng;
template <typename Type, int dimension>
struct FlexFlood {
    using T = std::array<Type, dimension>;
    using U = std::array<int, dimension>;
    using Iter = typename std::vector<T>::iterator;
    static constexpr double eps = 1e-9;
    std::vector<btree::btree_set<std::pair<double, T>>*> data;
    int num_data;
    U divideSize, divideSize_ini;
    int divideSizeSigma;
    std::vector<std::vector<double>> grid;
    std::vector<std::vector<int>> count_data;
    int sortDimension;
    FlexFlood(const Iter& begin, const Iter& end, const std::vector<std::vector<T>>& sample_queries)
        : data(0), num_data(end - begin), divideSize(), divideSizeSigma(0), grid(dimension), count_data(dimension) {
        learn(begin, end, sample_queries);
    }
    FlexFlood(const Iter& begin, const Iter& end, const U& divide, const int& sort_dim)
        : data(0), num_data(end - begin), divideSize(divide), divideSizeSigma(0), grid(dimension), count_data(dimension) {
        assert(divide[sort_dim] == 1);
        init(begin, end, divideSize, sort_dim);
    }
    inline void clear() {
        for(int i = 0; i < divideSizeSigma; ++i) {
            delete data[i];
        }
        data.resize(0);
        divideSize = {};
        divideSizeSigma = 0;
        for(int i = 0; i < dimension; ++i) grid[i].clear();
        for(int i = 0; i < dimension; ++i) count_data[i].clear();
    }
    void learn(const Iter& begin, const Iter& end, const std::vector<std::vector<T>>& sample_queries) {
        const int num_query = sample_queries.size();
        std::vector<std::vector<FeatureType>> divides;
        std::vector<AnswerType> weights_p, weights_r, weights_s;
        for(int sortdim = 0; sortdim < dimension; ++sortdim) {
            for(int i = 0; i < 31; ++i) {
                std::vector<FeatureType> divide(dimension, 0);
                U divsize = {};
                for(int d = 0; d < dimension; ++d) {
                    if(d == sortdim) divsize[d] = 1;
                    else divsize[d] = rng.random(2, 100);
                    divide[d] = divsize[d];
                }
                if(i > 0) divides.push_back(divide);
                init(begin, end, divsize, sortdim);
                std::tuple<AnswerType, AnswerType, AnswerType> weights = measure_time(num_query, sample_queries);
                if(i > 0) weights_p.push_back(std::get<0>(weights));
                if(i > 0) weights_r.push_back(std::get<1>(weights));
                if(i > 0) weights_s.push_back(std::get<2>(weights));
                clear();
            }
        }
        const int treesNo = 100, minNodeSize = 1, maxLevel = 50, numRandomFeatures = 3, numRandomPositions = 15;
        RandomForestRegression* rf_p = new RandomForestRegression(treesNo, minNodeSize, maxLevel, numRandomFeatures, numRandomPositions);
        RandomForestRegression* rf_r = new RandomForestRegression(treesNo, minNodeSize, maxLevel, numRandomFeatures, numRandomPositions);
        RandomForestRegression* rf_s = new RandomForestRegression(treesNo, minNodeSize, maxLevel, numRandomFeatures, numRandomPositions);
        rf_p->train(divides, weights_p);
        rf_r->train(divides, weights_r);
        rf_s->train(divides, weights_s);
        std::vector<FeatureType> best_divide(dimension);
        AnswerType min_weight = std::numeric_limits<AnswerType>::max();
        for(int sortdim = 0; sortdim < dimension; ++sortdim) {
            const int num_start = 5;
            for(int _ = 0; _ < num_start; ++_) {
                std::vector<FeatureType> divide(dimension, 0);
                for(int d = 0; d < dimension; ++d) {
                    divide[d] = rng.random(2, 100);
                }
                divide[sortdim] = 1;
                AnswerType weight = climing(divide, rf_p, rf_r, rf_s);
                if(weight < min_weight) {
                    min_weight = weight;
                    best_divide = divide;
                }
            }
        }
        for(int d = 0; d < dimension; ++d) {
            if(best_divide[d] == 1) sortDimension = d;
            divideSize[d] = best_divide[d];
        }
        init(begin, end, divideSize, sortDimension);
        delete rf_p;
        delete rf_r;
        delete rf_s;
    }
    inline AnswerType calc_weight(const std::vector<FeatureType>& divide, RandomForestRegression*& rf_p, RandomForestRegression*& rf_r, RandomForestRegression*& rf_s) {
        const AnswerType w_p = rf_p->estimateRegression(divide);
        const AnswerType w_r = rf_r->estimateRegression(divide);
        const AnswerType w_s = rf_s->estimateRegression(divide);
        return w_p + w_r + w_s;
    }
    inline AnswerType climing(std::vector<FeatureType>& divide, RandomForestRegression*& rf_p, RandomForestRegression*& rf_r, RandomForestRegression*& rf_s) {
        const double start_temp = 100000, end_temp = 10;
        const int TimeLimit = 2 * CLOCKS_PER_SEC;
        const int start_time = clock();
        AnswerType best_weight = calc_weight(divide, rf_p, rf_r, rf_s);
        int cnt = 0;
        while((int)clock() - start_time < TimeLimit) {
            cnt++;
            std::vector<FeatureType> cand_divide = divide;
            for(int d = 0; d < dimension; ++d) {
                if(cand_divide[d] == 1) continue;
                cand_divide[d] = std::clamp(cand_divide[d] + rng.random(-3, 4), 2, 100);
            }
            AnswerType cand_weight = calc_weight(cand_divide, rf_p, rf_r, rf_s);
            const double temp = start_temp + (end_temp - start_temp) * ((int)clock() - start_time) / TimeLimit;
            const double prob = exp((best_weight - cand_weight) / temp);
            if(prob > rrng(0.0, 1.0)) {
                best_weight = cand_weight;
                divide = cand_divide;
            }
        }
        return best_weight;
    }
    void init(const Iter& begin, const Iter& end, const U& divsize, const int sortdim) {
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
                assert(0 <= num_data * j / divsize[i] - 1 and num_data * j / divsize[i] - 1 < num_data);
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
                int idx = upper_bound(grid[i].begin(), grid[i].end(), (*iter)[i]) - grid[i].begin();  // *iterがi次元目でどこのグリッドに入る？
                --idx;
                assert(0 <= idx and idx < divsize[i]);
                index += idx * remDim;
                ++count_data[i][idx];
            }
            assert(0 <= index and index < divideSizeSigma);
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
    inline void scan_enumerate(const T& lower, const T& upper, const typename btree::btree_set<std::pair<double, T>>::const_iterator& iter, std::vector<T>& res) {
        bool flag = true;
        for(int j = 0; j < dimension; ++j) {
            if(!(lower[j] <= (*iter).second[j] and (*iter).second[j] <= upper[j])) {
                flag = false;
                break;
            }
        }
        if(flag) {
            res.push_back((*iter).second);
        }
    }
    inline void scan_count(const T& lower, const T& upper, const typename btree::btree_set<std::pair<double, T>>::const_iterator& iter, int& res) {
        bool flag = true;
        for(int j = 0; j < dimension; ++j) {
            if(!(lower[j] <= (*iter).second[j] and (*iter).second[j] <= upper[j])) {
                flag = false;
                break;
            }
        }
        if(flag) {
            ++res;
        }
    }
    std::vector<T> enumerate(const T& lower, const T& upper) {
        std::vector<T> res;
        std::vector<int> idx = projection(lower, upper);
        for(const int i : idx) {
            std::pair<typename btree::btree_set<std::pair<double, T>>::iterator, typename btree::btree_set<std::pair<double, T>>::iterator> seg = refinement(lower, upper, i);
            for(auto iter = seg.first; iter != seg.second; ++iter) {
                scan_enumerate(lower, upper, iter, res);
            }
        }
        return res;
    }
    int count(const T& lower, const T& upper) {
        int res = 0;
        std::vector<int> idx = projection(lower, upper);
        for(const int i : idx) {
            std::pair<typename btree::btree_set<std::pair<double, T>>::iterator, typename btree::btree_set<std::pair<double, T>>::iterator> seg = refinement(lower, upper, i);
            for(auto iter = seg.first; iter != seg.second; ++iter) {
                scan_count(lower, upper, iter, res);
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
            assert(0 <= idx[i] and idx[i] < divideSize[i]);
            index += idx[i] * sigma;
            sigma *= divideSize[i];
        }
        assert(0 <= index and index < divideSizeSigma);
        data[index]->insert(make_pair(dat[sortDimension], dat));
        ++num_data;
        for(int i = 0; i < dimension; ++i) {
            ++count_data[i][idx[i]];
            if(lower.first * count_data[i][idx[i]] * divideSize_ini[i] <= lower.second * num_data) {
                if(idx[i] == 0) {
                    if(middle.first * count_data[i][idx[i] + 1] * divideSize_ini[i] <= middle.second * num_data) {
                        merge(i, idx[i]);
                    } else {
                        flatten(i, idx[i]);
                    }
                } else {
                    if(middle.first * count_data[i][idx[i] - 1] * divideSize_ini[i] <= middle.second * num_data) {
                        merge(i, idx[i] - 1);
                    } else {
                        flatten(i, idx[i] - 1);
                    }
                }
            } else if(upper.first * count_data[i][idx[i]] * divideSize_ini[i] >= upper.second * num_data) {
                split(i, idx[i]);
            }
        }
    }
    void insert_only(const T& dat) {
        int index = 0, sigma = 1;
        for(int i = dimension - 1; i >= 0; --i) {
            if(i == sortDimension) continue;
            int idx = upper_bound(grid[i].begin(), grid[i].end(), dat[i]) - grid[i].begin() - 1;
            assert(0 <= idx and idx < divideSize[i]);
            index += idx * sigma;
            sigma *= divideSize[i];
        }
        assert(0 <= index and index < divideSizeSigma);
        data[index]->insert(make_pair(dat[sortDimension], dat));
    }
    void erase(const T& dat, const std::pair<int, int>& lower = std::make_pair(3, 1), const std::pair<int, int>& upper = std::make_pair(1, 2)) {
        const std::pair<int, int> middle = {lower.first * upper.first, lower.second * upper.first + upper.second * lower.first};
        std::vector<int> idx(dimension);
        int index = 0, sigma = 1;
        for(int i = dimension - 1; i >= 0; --i) {
            if(i == sortDimension) continue;
            idx[i] = upper_bound(grid[i].begin(), grid[i].end(), dat[i]) - grid[i].begin() - 1;
            assert(0 <= idx[i] and idx[i] < divideSize[i]);
            index += idx[i] * sigma;
            sigma *= divideSize[i];
        }
        assert(0 <= index and index < divideSizeSigma);
        data[index]->erase(make_pair(dat[sortDimension], dat));
        --num_data;
        for(int i = 0; i < dimension; ++i) {
            --count_data[i][idx[i]];
            if(lower.first * count_data[i][idx[i]] * divideSize_ini[i] <= lower.second * num_data) {
                if(idx[i] == 0) {
                    if(middle.first * count_data[i][idx[i] + 1] * divideSize_ini[i] <= middle.second * num_data) {
                        merge(i, idx[i]);
                    } else {
                        flatten(i, idx[i]);
                    }
                } else {
                    if(middle.first * count_data[i][idx[i] - 1] * divideSize_ini[i] <= middle.second * num_data) {
                        merge(i, idx[i] - 1);
                    } else {
                        flatten(i, idx[i] - 1);
                    }
                }
            } else if(upper.first * count_data[i][idx[i]] * divideSize_ini[i] >= upper.second * num_data) {
                split(i, idx[i]);
            }
        }
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
    void flatten(const int dim, const int idx) {
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
    std::tuple<int, int, int> measure_time(const int& num_query, const std::vector<std::vector<T>>& queries) {
        AnswerType w_p = 0, w_r = 0, w_s = 0;
        std::chrono::system_clock::time_point start, end;
        std::chrono::nanoseconds time;
        std::chrono::_V2::steady_clock::rep msec;
        int res = 0;
        for(int q = 0; q < num_query; ++q) {
            start = std::chrono::system_clock::now();
            std::vector<int> idx = projection(queries[q][0], queries[q][1]);
            end = std::chrono::system_clock::now();
            time = end - start;
            msec = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
            w_p += msec;
            std::vector<std::pair<typename btree::btree_set<std::pair<double, T>>::iterator, typename btree::btree_set<std::pair<double, T>>::iterator>> seg((int)idx.size());
            start = std::chrono::system_clock::now();
            for(int i = 0; i < (int)idx.size(); ++i) {
                seg[i] = refinement(queries[q][0], queries[q][1], idx[i]);
            }
            end = std::chrono::system_clock::now();
            time = end - start;
            msec = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
            w_r += msec;
            start = std::chrono::system_clock::now();
            for(int i = 0; i < (int)seg.size(); ++i) {
                for(auto iter = seg[i].first; iter != seg[i].second; ++iter) {
                    scan_count(queries[q][0], queries[q][1], iter, res);
                }
            }
            end = std::chrono::system_clock::now();
            time = end - start;
            msec = std::chrono::duration_cast<std::chrono::microseconds>(time).count();
            w_s += msec;
        }
        std::cout << res << '\n';
        return std::make_tuple(w_p, w_r, w_s);
    }
};