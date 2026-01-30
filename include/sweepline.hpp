/**
 * @file sweepline.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Rectangles sweep line algorithm using segment tree.
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace gmor {

template <typename Scalar> struct Segment {

    // Segment stores y before sweeping, and then transform to sorted integer index y_i for query in the tree.
    union {
        Scalar yl;
        uint32_t yl_i{};
    };

    union {
        Scalar yr;
        uint32_t yr_i{};
    };

    Scalar x, weight;

    Segment() = default;

    Segment(Scalar yl_, Scalar yr_, Scalar x_, Scalar w_) : yl(yl_), yr(yr_), x(x_), weight(w_) {}

    ~Segment() = default;

    bool operator<(const Segment& other) const { return (x < other.x) || (!(other.x < x) && weight < other.weight); }
};

template <typename Scalar> using Segments = std::vector<Segment<Scalar>>;

template <typename Scalar> struct SegmentNode {
    uint32_t l, r;
    Scalar val, lazy;
};

/**
 * @brief Recursive implementation of segment tree.
 *
 * @tparam Scalar Type
 *
 */
template <typename Scalar> class SegmentTree {
  public:
    explicit SegmentTree(uint32_t max_n);
    ~SegmentTree();

    // Initialize the segment tree
    void build(uint32_t n);
    // Recursive build
    void build(uint32_t p, uint32_t l, uint32_t r);

    // Update the maximum overlapping with child nodes
    void pushup(uint32_t p);

    // Push lazy propagation to child nodes
    void pushdown(uint32_t p);

    // Recursive update the value of closed interval [l, r] with val
    void update(uint32_t l, uint32_t r, Scalar val, uint32_t p = 1);

    // Query the maximum overlapping of closed interval [l, r] in the tree
    Scalar query(uint32_t l, uint32_t r, uint32_t p = 1);

    // Get maximum overlapping of root
    inline Scalar getMax() { return data[1].val; }

  protected:
    // Segment tree rooted with 1
    std::vector<SegmentNode<Scalar>> data;
};

/**
 * @brief Non-recursive implementation with ZKW tree referenced from Kunwei Zhang's slides of segment tree.
 *
 * @tparam Scalar Type
 *
 * An efficient implementation that returns the same result as the recursive version.
 */
template <typename Scalar> class SegmentTreeZKW {
  public:
    explicit SegmentTreeZKW(uint32_t max_n);
    ~SegmentTreeZKW();

    // Initialize the segment tree
    void build(uint32_t n);

    // Update the value of closed interval [l, r] with val
    void update(uint32_t l, uint32_t r, Scalar val);

    // Query the maximum overlapping of closed interval [l, r] in the tree
    Scalar query(uint32_t l, uint32_t r);

    // Get maximum overlapping of root
    inline Scalar getMax() { return data[1]; }

  protected:
    // Complete Binary Tree rooted with 1, and each node stores val[n] - val[n >> 1].
    std::vector<Scalar> data;
    uint32_t N;
};

// Sweep line algorithm for maximum overlapping of 2D rectangles using 1D segment tree.
template <typename Scalar, class SegTree> Scalar sweepLine2D(Segments<Scalar>& segs, SegTree& tree);

/*****************Implementation******************/

/*****************SegmentTree******************/

template <typename Scalar> SegmentTree<Scalar>::SegmentTree(uint32_t max_n) : data(max_n << 2) {}

template <typename Scalar> SegmentTree<Scalar>::~SegmentTree() = default;

template <typename Scalar> void SegmentTree<Scalar>::build(uint32_t n) { build(1, 1, n); }

template <typename Scalar> void SegmentTree<Scalar>::build(uint32_t p, uint32_t l, uint32_t r) {
    data[p].l = l;
    data[p].r = r;
    data[p].val = data[p].lazy = 0;
    if (l == r)
        return;
    uint32_t mid = (l + r) >> 1;
    build(p << 1, l, mid);
    build(p << 1 | 1, mid + 1, r);
}

template <typename Scalar> void SegmentTree<Scalar>::pushup(uint32_t p) {
    data[p].val = std::max(data[p << 1].val, data[p << 1 | 1].val);
}

template <typename Scalar> void SegmentTree<Scalar>::pushdown(uint32_t p) {
    data[p << 1].val += data[p].lazy;
    data[p << 1 | 1].val += data[p].lazy;
    data[p << 1].lazy += data[p].lazy;
    data[p << 1 | 1].lazy += data[p].lazy;
    data[p].lazy = 0; // Clear lazy
}

template <typename Scalar> void SegmentTree<Scalar>::update(uint32_t l, uint32_t r, Scalar val, uint32_t p) {
    if (l <= data[p].l && data[p].r <= r) {
        data[p].val += val;
        data[p].lazy += val;
        return;
    }
    pushdown(p);
    uint32_t mid = (data[p].l + data[p].r) >> 1;
    if (l <= mid)
        update(p << 1, l, r, val);
    if (r > mid)
        update(p << 1 | 1, l, r, val);
    pushup(p);
}

template <typename Scalar> Scalar SegmentTree<Scalar>::query(uint32_t l, uint32_t r, uint32_t p) {
    if (l <= data[p].l && r >= data[p].r)
        return data[p].val;
    pushdown(p);
    Scalar val = 0;
    uint32_t mid = (data[p].l + data[p].r) >> 1;
    if (l <= mid)
        val = query(p << 1, l, r);
    if (r > mid)
        val = std::max(val, query(p << 1 | 1, l, r));
    return val;
}

/*****************SegmentTree end******************/

/*****************SegmentTreeZKW******************/

template <typename Scalar> SegmentTreeZKW<Scalar>::SegmentTreeZKW(uint32_t max_n) {
    for (N = 1; N < max_n + 2; N <<= 1)
        ;
    data.resize(N + max_n + 3);
}

template <typename Scalar> SegmentTreeZKW<Scalar>::~SegmentTreeZKW() = default;

template <typename Scalar> void SegmentTreeZKW<Scalar>::build(uint32_t n) {
    // Add two sentinel nodes
    for (N = 1; N < n + 2; N <<= 1)
        ;
    std::fill(data.begin() + 1, data.begin() + N + n + 3, 0);
}

template <typename Scalar> void SegmentTreeZKW<Scalar>::update(uint32_t l, uint32_t r, Scalar val) {
    Scalar tmp;
    // Start with two sentinel nodes (Open interval), and access the parent nodes until l and r are sibling nodes
    for (l += N - 1, r += N + 1; l ^ r ^ 1; l >>= 1, r >>= 1) {
        // If left sentinel is left leaf, the right subtree is fully covered, equivalent to lazy propagation
        // The children do not need to be update due to data[n] = val[n] - val[n >> 1]
        if (~l & 1)
            data[l ^ 1] += val;
        if (r & 1)
            data[r ^ 1] += val;
        // Pushup maximum overlapping of sibling nodes to parent
        tmp = std::max(data[l], data[l ^ 1]);
        data[l] -= tmp;
        data[l ^ 1] -= tmp;
        data[l >> 1] += tmp;
        tmp = std::max(data[r], data[r ^ 1]);
        data[r] -= tmp;
        data[r ^ 1] -= tmp;
        data[r >> 1] += tmp;
    }

    // Accumulate to root
    while (l > 1) {
        tmp = std::max(data[l], data[l ^ 1]);
        data[l] -= tmp;
        data[l ^ 1] -= tmp;
        data[l >>= 1] += tmp;
    }
}

template <typename Scalar> Scalar SegmentTreeZKW<Scalar>::query(uint32_t l, uint32_t r) {
    Scalar lAns = 0, rAns = 0;
    // Closed interval. Make sure l < r, if l == r the loop will be endless
    // But why using open interval that returns RMQ of [l - 1, r + 1] in original slides?
    if (l < r) {
        for (l += N, r += N; l ^ r ^ 1; l >>= 1, r >>= 1) {
            lAns += data[l], rAns += data[r];
            if (~l & 1)
                lAns = std::max(lAns, data[l ^ 1]);
            if (r & 1)
                rAns = std::max(rAns, data[r ^ 1]);
        }
        lAns = std::max(lAns, rAns);
    } else {
        // l >= r, return l node
        lAns = data[l += N];
    }

    // Accumulate to root
    while (l > 1)
        lAns += data[l >>= 1];
    return lAns;
}

/*****************SegmentTreeZKW end******************/

/*****************sweepLine2D******************/

template <typename Scalar, class SegTree> Scalar sweepLine2D(Segments<Scalar>& segs, SegTree& tree) {
    uint32_t num_segs = segs.size() - 1;
    std::vector<uint32_t> indices(num_segs + 1);
    std::iota(indices.begin() + 1, indices.end(), 1);
    std::sort(indices.begin() + 1, indices.end(), [&](const auto& i, const auto& j) {
        return (i & 1 ? segs[i].yl : segs[i].yr) < (j & 1 ? segs[j].yl : segs[j].yr);
    });

    for (uint32_t i = 1; i <= num_segs; ++i) {
        if (indices[i] & 1) {
            segs[indices[i]].yl_i = segs[indices[i] + 1].yl_i = i;
        } else {
            segs[indices[i] - 1].yr_i = segs[indices[i]].yr_i = i;
        }
    }

    std::sort(segs.begin() + 1, segs.end());
    tree.build(num_segs);
    Scalar max_val = 0.0;
    for (uint32_t i = 1; i <= num_segs; ++i) {
        tree.update(segs[i].yl_i, segs[i].yr_i, segs[i].weight);
        max_val = std::max(max_val, tree.getMax());
    }
    return max_val;
}

/*****************sweepLine2D end******************/

} // namespace gmor
