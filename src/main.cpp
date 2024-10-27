#include <bits/stdc++.h>
#include "FlexFlood.hpp"
int main(void) {
    using namespace std;
    using Type = double;
    constexpr int dimension = 3;
    using T = std::array<Type, dimension>;
    cin.tie(0);
    ios::sync_with_stdio(0);
    cout << fixed << setprecision(20);
    int n;
    cin >> n;
    vector<T> p(n);
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < dimension; ++j) {
            cin >> p[i][j];
        }
    }
    int sq;
    cin >> sq;
    vector<vector<T>> sample_queries(sq, vector<T>(2));
    for(int i = 0; i < sq; ++i) {
        int type;
        cin >> type;
        if(type == 0 or type == 1) {
            for(int j = 0; j < dimension; ++j) {
                cin >> sample_queries[i][0][j];
            }
            for(int j = 0; j < dimension; ++j) {
                cin >> sample_queries[i][1][j];
            }
        } else {
            assert(0);
        }
    }
    int q;
    cin >> q;
    vector<int> type(q);
    vector<T> lower(q), upper(q), dat(q);
    for(int i = 0; i < q; ++i) {
        cin >> type[i];
        if(type[i] == 0) {
            for(int j = 0; j < dimension; ++j) {
                cin >> lower[i][j];
            }
            for(int j = 0; j < dimension; ++j) {
                cin >> upper[i][j];
            }
        } else if(type[i] == 1) {
            for(int j = 0; j < dimension; ++j) {
                cin >> lower[i][j];
            }
            for(int j = 0; j < dimension; ++j) {
                cin >> upper[i][j];
            }
        } else if(type[i] == 2) {
            for(int j = 0; j < dimension; ++j) {
                cin >> dat[i][j];
            }
        } else if(type[i] == 3) {
            for(int j = 0; j < dimension; ++j) {
                cin >> dat[i][j];
            }
        } else {
            assert(0);
        }
    }
    long long sum = 0;
    FlexFlood<Type, dimension> index(p.begin(), p.end(), sample_queries);
    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
    for(int i = 0; i < q; ++i) {
        if(type[i] == 0) {
            vector<T> ans = index.enumerate(lower[i], upper[i]);
            sum += ans.size();
        } else if(type[i] == 1) {
            vector<T> ans = index.enumerate(lower[i], upper[i]);
            sum += ans.size();
        } else if(type[i] == 2) {
            index.insert(dat[i]);
        } else if(type[i] == 3) {
            index.erase(dat[i]);
        } else {
            assert(0);
        }
        if((i + 1) % 10000 == 0) {
            cerr << i + 1 << '\n';
            end = chrono::system_clock::now();
            auto time = end - start;
            auto msec = chrono::duration_cast<chrono::milliseconds>(time).count();
            cerr << msec << " (ms)" << '\n';
        }
    }
    end = chrono::system_clock::now();
    auto time = end - start;
    auto msec = chrono::duration_cast<chrono::milliseconds>(time).count();
    cerr << msec << " (ms)" << '\n';
    cout << sum << '\n';
    for(int i = 0; i < dimension; ++i) {
        cerr << "dimension " << i << '\n';
        cerr << index.count_data[i].size() << '\n';
        int sum = 0;
        for(int j = 0; j < (int)index.count_data[i].size(); ++j) {
            sum += index.count_data[i][j];
        }
        cerr << sum << '\n';
    }
    index.clear();
    return 0;
}