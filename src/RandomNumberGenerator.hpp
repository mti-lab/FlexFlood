#pragma once
#include <random>
#include <chrono>
#include <algorithm>
struct RandomNumberGenerator {
    std::mt19937_64 mt;
    RandomNumberGenerator()
        : mt(std::chrono::steady_clock::now().time_since_epoch().count()) {}
    inline int operator()(int l, int r) {
        std::uniform_int_distribution<int> dist(l, r - 1);
        return dist(mt);
    }
} rng;
struct RealRandomNumberGenerator {
    std::mt19937_64 mt;
    RealRandomNumberGenerator()
        : mt(std::chrono::steady_clock::now().time_since_epoch().count()) {}
    inline double operator()(double l, double r) {
        std::uniform_real_distribution<double> dist(l, r);
        return dist(mt);
    }
} rrng;
struct Normal {
    std::mt19937_64 mt;
    Normal()
        : mt(std::chrono::steady_clock::now().time_since_epoch().count()) {}
    inline double operator()(int u, int s) {
        std::normal_distribution<> dist(u, s);
        return std::clamp(dist(mt), 0.0, 1e9);
    }
} normal;