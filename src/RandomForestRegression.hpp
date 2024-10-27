#pragma once
#include <bits/stdc++.h>
#define SZ(a) ((int)(a).size())
struct RandomNumberGenerator {
    std::mt19937_64 mt;
    RandomNumberGenerator()
        : mt(std::chrono::steady_clock::now().time_since_epoch().count()) {}
    inline int random(const int l = 0, const int r = std::numeric_limits<int>::max()) {
        std::uniform_int_distribution<int> dist(l, r - 1);
        return dist(mt);
    }
    inline void randomShuffle(std::vector<int>& a) {
        const int n = SZ(a);
        for(int i = n - 1; i > 0; --i) {
            std::swap(a[i], a[random(0, i + 1)]);
        }
    }
} rng;
typedef int FeatureType;
typedef int AnswerType;
struct TreeNode {
    bool leaf;
    int level;
    int featureID;
    FeatureType value;
    AnswerType answer;
    std::vector<int> bags;
    int left;
    int right;

    TreeNode() {
        leaf = false;
        level = -1;
        featureID = -1;
        value = 0;
        answer = 0;
        left = -1;
        right = -1;
    }
};
class DecisionTree {
   public:
    DecisionTree() {}
    DecisionTree(const std::vector<std::vector<FeatureType> >& features,
                 const std::vector<AnswerType>& answers,
                 int minNodeSize,
                 int maxLevel,
                 int numRandomFeatures,
                 int numRandomPositions) {
        const int numData = SZ(features);
        const int numFeatures = SZ(features[0]);
        assert(numData == SZ(answers));
        assert(numData > 1);
        TreeNode root;
        root.level = 0;
        root.bags.resize(numData);
        for(int i = 0; i < numData; ++i) {
            root.bags[i] = rng.random(0, numData);
        }
        m_nodes.emplace_back(root);
        int curNode = 0;
        while(curNode < SZ(m_nodes)) {
            TreeNode& node = m_nodes[curNode];
            bool equal = true;
            for(int i = 1; i < SZ(node.bags); ++i) {
                if(answers[node.bags[i]] != answers[node.bags[i - 1]]) {
                    equal = false;
                    break;
                }
            }
            if(equal || SZ(node.bags) <= minNodeSize || node.level >= maxLevel) {
                setLeaf(node, curNode, answers);
                continue;
            }
            int bestFeatureID = -1;
            int bestLeft = 0, bestRight = 0;
            FeatureType bestValue = 0;
            double bestMSE = 1e99;

            for(int i = 0; i < numRandomFeatures; ++i) {
                const int featureID = rng.random(0, numFeatures);
                for(int j = 0; j < numRandomPositions; ++j) {
                    const int positionID = rng.random(0, SZ(node.bags));
                    const FeatureType splitValue = features[node.bags[positionID]][featureID];
                    double sumLeft = 0;
                    double sumRight = 0;
                    int totalLeft = 0;
                    int totalRight = 0;
                    for(int p : node.bags) {
                        if(features[p][featureID] < splitValue) {
                            sumLeft += answers[p];
                            totalLeft++;
                        } else {
                            sumRight += answers[p];
                            totalRight++;
                        }
                    }
                    if(totalLeft == 0 || totalRight == 0) continue;
                    double meanLeft = totalLeft == 0 ? 0 : sumLeft / totalLeft;
                    double meanRight = totalRight == 0 ? 0 : sumRight / totalRight;
                    double mse = 0;
                    for(int p : node.bags) {
                        if(features[p][featureID] < splitValue) {
                            mse += (answers[p] - meanLeft) * (answers[p] - meanLeft);
                        } else {
                            mse += (answers[p] - meanRight) * (answers[p] - meanRight);
                        }
                    }
                    if(mse < bestMSE) {
                        bestMSE = mse;
                        bestValue = splitValue;
                        bestFeatureID = featureID;
                        bestLeft = totalLeft;
                        bestRight = totalRight;
                    }
                }
            }
            if(bestLeft == 0 || bestRight == 0) {
                setLeaf(node, curNode, answers);
                continue;
            }
            TreeNode left;
            TreeNode right;
            left.level = right.level = node.level + 1;
            node.featureID = bestFeatureID;
            node.value = bestValue;
            node.left = SZ(m_nodes);
            node.right = SZ(m_nodes) + 1;
            left.bags.resize(bestLeft);
            right.bags.resize(bestRight);
            for(int p : node.bags) {
                if(features[p][node.featureID] < node.value) {
                    left.bags[--bestLeft] = p;
                } else {
                    right.bags[--bestRight] = p;
                }
            }
            m_nodes.emplace_back(left);
            m_nodes.emplace_back(right);
            curNode++;
        }
    }
    AnswerType estimate(const std::vector<FeatureType>& features) const {
        const TreeNode* pNode = &m_nodes[0];
        while(true) {
            if(pNode->leaf) {
                break;
            }
            const int nextNodeID = features[pNode->featureID] < pNode->value ? pNode->left : pNode->right;
            pNode = &m_nodes[nextNodeID];
        }
        return pNode->answer;
    }

   private:
    void setLeaf(TreeNode& node, int& curNode, const std::vector<AnswerType>& answers) const {
        node.leaf = true;
        for(int p : node.bags) {
            node.answer += answers[p];
        }
        assert(SZ(node.bags) > 0);
        if(SZ(node.bags)) {
            node.answer /= SZ(node.bags);
        }
        curNode++;
    }
    std::vector<TreeNode> m_nodes;
};

class RandomForestRegression {
   public:
    RandomForestRegression(const int _treesNo = 100, const int _minNodeSize = 1, const int _maxLevel = 20, const int _numRandomFeatures = 2, const int _numRandomPositions = 10) {
        treesNo = _treesNo, minNodeSize = _minNodeSize, maxLevel = _maxLevel, numRandomFeatures = _numRandomFeatures, numRandomPositions = _numRandomPositions;
        clear();
    }

    void clear() {
        m_trees.clear();
    }
    void train(const std::vector<std::vector<FeatureType> >& features, const std::vector<AnswerType>& answers) {
        for(int i = 0; i < treesNo; ++i) {
            m_trees.emplace_back(DecisionTree(features, answers, minNodeSize, maxLevel, numRandomFeatures, numRandomPositions));
        }
    }
    AnswerType estimateRegression(const std::vector<FeatureType>& features) {
        if(SZ(m_trees) == 0) {
            return 0.0;
        }
        double sum = 0;
        for(int i = 0; i < SZ(m_trees); ++i) {
            sum += m_trees[i].estimate(features);
        }
        return sum / SZ(m_trees);
    }

   private:
    int treesNo, minNodeSize, maxLevel, numRandomFeatures, numRandomPositions;
    std::vector<DecisionTree> m_trees;
};
