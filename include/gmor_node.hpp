/**
 * @file gmor_node.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Node of GMOR
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include "interval_stabbing.hpp"
#include "registration_base.hpp"
#include "sphere_projection.hpp"
#include "sweepline.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace gmor {

template <typename Scalar, class Projection = SphereProjCube<Scalar>>
class Node2Stage1 : public Node<Scalar, Node2Stage1<Scalar, Projection>> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<Scalar, 2, 2> region;
    Scalar d[2];

    Node2Stage1(Scalar x_l, Scalar x_r, Scalar y_l, Scalar y_r, const Projection& projection, Scalar d_l = -1e6,
                Scalar d_r = 1e6);
    Node2Stage1(const Eigen::Matrix<Scalar, 2, 2>& region_lr, const Projection& projection, Scalar d_l = -1e6,
                Scalar d_r = 1e6);
    Node2Stage1(int k, int depth, Scalar d_l = -1e6, Scalar d_r = 1e6);
    ~Node2Stage1() = default;

    std::vector<Node2Stage1<Scalar, Projection>, Eigen::aligned_allocator<Node2Stage1<Scalar, Projection>>> split();

    Scalar volume() const;

    void estULB(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& source,
                const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& target, const std::vector<Scalar>& weights,
                std::array<Intervals<Scalar, Scalar>, 2>& intervals, Scalar xi, Scalar best_lb, Scalar rho);

    Eigen::Vector<Scalar, 3> getAxis();

    using gmor::Node<Scalar, Node2Stage1<Scalar, Projection>>::lb;
    using gmor::Node<Scalar, Node2Stage1<Scalar, Projection>>::ub;
    using Ptr = std::shared_ptr<Node2Stage1<Scalar, Projection>>;
    using Ptr_u = std::unique_ptr<Node2Stage1<Scalar, Projection>>;

  protected:
    Projection m_projection;
};

template <typename Scalar> class NodeStage2 : public Node<Scalar, NodeStage2<Scalar>> {
  public:
    Scalar theta[2]; // 1 DOF of rotation angle

    NodeStage2(Scalar theta_l, Scalar theta_r);
    ~NodeStage2() = default;

    std::vector<NodeStage2<Scalar>> split();

    Scalar volume() const;

    void estUB(const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& midpoints,
               const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& vecs, const std::vector<Scalar>& weights,
               Segments<Scalar>& segments, SegmentTreeZKW<Scalar>& segtree, Scalar xi);

    void estLB(const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& midpoints,
               const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& vecs, const std::vector<Scalar>& weights,
               Segments<Scalar>& segments, SegmentTreeZKW<Scalar>& segtree, Scalar xi);

    using gmor::Node<Scalar, NodeStage2<Scalar>>::lb;
    using gmor::Node<Scalar, NodeStage2<Scalar>>::ub;
    using Ptr = std::shared_ptr<NodeStage2<Scalar>>;
    using Ptr_u = std::unique_ptr<NodeStage2<Scalar>>;
};

/*****************Implementation******************/

/*****************Node2Stage1******************/
template <typename Scalar, class Projection>
Node2Stage1<Scalar, Projection>::Node2Stage1(Scalar x_l, Scalar x_r, Scalar y_l, Scalar y_r,
                                             const Projection& projection, Scalar d_l, Scalar d_r)
    : Node<Scalar, Node2Stage1<Scalar, Projection>>(), d{d_l, d_r}, m_projection(projection) {
    region << x_l, x_r, y_l, y_r;
}

template <typename Scalar, class Projection>
Node2Stage1<Scalar, Projection>::Node2Stage1(const Eigen::Matrix<Scalar, 2, 2>& region_lr, const Projection& projection,
                                             Scalar d_l, Scalar d_r)
    : Node<Scalar, Node2Stage1<Scalar, Projection>>(), region(region_lr), d{d_l, d_r}, m_projection(projection) {}

template <typename Scalar, class Projection>
Node2Stage1<Scalar, Projection>::Node2Stage1(int k, int depth, Scalar d_l, Scalar d_r)
    : Node<Scalar, Node2Stage1<Scalar, Projection>>(), d{d_l, d_r} {
    region = m_projection.initRegion(k, depth);
}

template <typename Scalar, class Projection>
std::vector<Node2Stage1<Scalar, Projection>, Eigen::aligned_allocator<Node2Stage1<Scalar, Projection>>>
Node2Stage1<Scalar, Projection>::split() {
    std::vector<Node2Stage1<Scalar, Projection>, Eigen::aligned_allocator<Node2Stage1<Scalar, Projection>>> nodes;
    // Binary splitting, find the dimension with the largest range and split into two nodes
    if (region(0, 1) - region(0, 0) > region(1, 1) - region(1, 0)) {
        nodes.emplace_back(region(0, 0), 0.5 * (region(0, 0) + region(0, 1)), region(1, 0), region(1, 1),
                           this->m_projection, d[0], d[1]);
        nodes.emplace_back(0.5 * (region(0, 0) + region(0, 1)), region(0, 1), region(1, 0), region(1, 1),
                           this->m_projection, d[0], d[1]);
    } else {
        nodes.emplace_back(region(0, 0), region(0, 1), region(1, 0), 0.5 * (region(1, 0) + region(1, 1)),
                           this->m_projection, d[0], d[1]);
        nodes.emplace_back(region(0, 0), region(0, 1), 0.5 * (region(1, 0) + region(1, 1)), region(1, 1),
                           this->m_projection, d[0], d[1]);
    }
    return nodes;
}

template <typename Scalar, class Projection>
void Node2Stage1<Scalar, Projection>::estULB(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& source,
                                             const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& target,
                                             const std::vector<Scalar>& weights,
                                             std::array<Intervals<Scalar, Scalar>, 2>& intervals, Scalar xi,
                                             Scalar best_lb, Scalar rho) {
    if (source.cols() < 3 || source.cols() != target.cols() || source.cols() != (int)weights.size()) {
        std::cerr << "Input size error" << std::endl;
        return;
    }

    Eigen::Vector<Scalar, 3> r_c = m_projection.projInvRegionCenter(region);
    Eigen::Vector<Scalar, 3> r_max = m_projection.projInvRegionMax(region);
    Scalar tau_cos = r_c.dot(r_max);
    Scalar tau_sin = std::sqrt(1.0 - tau_cos * tau_cos);

    intervals[0].clear();
    intervals[1].clear();
    for (int i = 0; i < source.cols(); ++i) {
        Eigen::Vector<Scalar, 3> xy = target.col(i) - source.col(i);
        // Projection of xy onto r_c
        Scalar d_proj = xy.dot(r_c);
        if (d[0] < d_proj && d_proj < d[1]) {
            Scalar xy_norm = xy.norm();
            intervals[0].emplace_back(d_proj, weights[i]);
            // Check the x to y vector is in the area of r_c
            Scalar d_vert = xy.cross(r_c).norm();
            Scalar interval_rise, interval_fall;
            if (d_proj > 0) {
                interval_rise = d_proj * tau_cos - tau_sin * d_vert - xi;
                if (d_proj < tau_cos * xy_norm) { // cos between xy and r_c < cos between r_max and r_c
                    interval_fall = d_proj * tau_cos + tau_sin * d_vert + xi;
                } else {
                    interval_fall = xy_norm + xi;
                }
            } else { // in the opposite direction
                interval_fall = d_proj * tau_cos + tau_sin * d_vert + xi;
                if (-d_proj < tau_cos * xy_norm) {
                    interval_rise = d_proj * tau_cos - tau_sin * d_vert - xi;
                } else {
                    interval_rise = -xy_norm - xi;
                }
            }

            intervals[1].emplace_back(interval_rise, weights[i]);
            intervals[1].emplace_back(interval_fall, -weights[i]);
        }
    }
    ub = intervalStabbing<Scalar, Scalar>(intervals[1], d, best_lb, rho);
    // Prune current node
    if (ub <= best_lb) {
        lb = 0;
        return;
    }
    lb = intervalStabbingConstLen<Scalar, Scalar>(intervals[0], 2 * xi);

    // Generally unreachable but it might happen due to numerical error
    if (lb > ub) {
        ub = lb;
    }
}

template <typename Scalar, class Projection> Eigen::Vector<Scalar, 3> Node2Stage1<Scalar, Projection>::getAxis() {
    return m_projection.projInvRegionCenter(region);
}

template <typename Scalar, class Projection> Scalar Node2Stage1<Scalar, Projection>::volume() const {
    return (region(0, 1) - region(0, 0)) * (region(1, 1) - region(1, 0));
}
/*****************Node2Stage1 end******************/

/*****************NodeStage2******************/
template <typename Scalar>
NodeStage2<Scalar>::NodeStage2(Scalar theta_l, Scalar theta_r)
    : Node<Scalar, NodeStage2<Scalar>>(), theta{theta_l, theta_r} {}

template <typename Scalar> std::vector<NodeStage2<Scalar>> NodeStage2<Scalar>::split() {
    std::vector<NodeStage2<Scalar>> nodes;
    nodes.emplace_back(theta[0], 0.5 * (theta[0] + theta[1]));
    nodes.emplace_back(0.5 * (theta[0] + theta[1]), theta[1]);
    return nodes;
}

template <typename Scalar>
void NodeStage2<Scalar>::estUB(const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& midpoints,
                               const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& vecs, const std::vector<Scalar>& weights,
                               Segments<Scalar>& segments, SegmentTreeZKW<Scalar>& segtree, Scalar xi) {
    if (midpoints.cols() < 3 || midpoints.cols() != vecs.cols() || midpoints.cols() != (int)weights.size()) {
        std::cerr << "Input size error" << std::endl;
        return;
    }
    if ((int)segments.size() != ((midpoints.cols() << 1) | 1)) {
        segments.resize((midpoints.cols() << 1) | 1);
    }

    // theta/2 in the range of (0, pi)
    Scalar theta_cot_l = std::tan(M_PI_2 - 0.5 * theta[0]);
    Scalar theta_cot_r = std::tan(M_PI_2 - 0.5 * theta[1]);

    // Amplified error, csc^2(theta/2) = 1 + cot^2(theta/2)
    Scalar xi_l = xi * std::sqrt(1 + theta_cot_l * theta_cot_l);
    Scalar xi_r = xi * std::sqrt(1 + theta_cot_r * theta_cot_r);

    for (int i = 0, j = 1; i < midpoints.cols(); ++i, j += 2) {
        Eigen::Vector<Scalar, 2> rotcenter_l = midpoints.col(i) + vecs.col(i) * theta_cot_l;
        Eigen::Vector<Scalar, 2> rotcenter_r = midpoints.col(i) + vecs.col(i) * theta_cot_r;

        // Find the bounding box of rotation center for upper bound
        Scalar bounds[2][2];
        bounds[0][0] = std::min(rotcenter_l[0] - xi_l, rotcenter_r[0] - xi_r);
        bounds[0][1] = std::max(rotcenter_l[0] + xi_l, rotcenter_r[0] + xi_r);
        bounds[1][0] = std::min(rotcenter_l[1] - xi_l, rotcenter_r[1] - xi_r);
        bounds[1][1] = std::max(rotcenter_l[1] + xi_l, rotcenter_r[1] + xi_r);

        segments[j] = {bounds[1][0], bounds[1][1], bounds[0][0], weights[i]};
        segments[j + 1] = {bounds[1][0], bounds[1][1], bounds[0][1], -weights[i]};
    }

    ub = sweepLine2D(segments, segtree);
}

template <typename Scalar>
void NodeStage2<Scalar>::estLB(const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& midpoints,
                               const Eigen::Matrix<Scalar, 2, Eigen::Dynamic>& vecs, const std::vector<Scalar>& weights,
                               Segments<Scalar>& segments, SegmentTreeZKW<Scalar>& segtree, Scalar xi) {
    if (midpoints.cols() < 3 || midpoints.cols() != vecs.cols() || midpoints.cols() != (int)weights.size()) {
        std::cerr << "Input size error" << std::endl;
        return;
    }
    if ((int)segments.size() != ((midpoints.cols() << 1) | 1)) {
        segments.resize((midpoints.cols() << 1) | 1);
    }

    // theta/2 in the range of (0, pi)
    Scalar theta_cot_c = std::tan(M_PI_2 - 0.25 * (theta[0] + theta[1])); // cot(theta/2)

    // Amplified error, csc^2(theta/2) = 1 + cot^2(theta/2)
    Scalar xi_c = xi * std::sqrt(1 + theta_cot_c * theta_cot_c);

    for (int i = 0, j = 1; i < midpoints.cols(); ++i, j += 2) {
        Eigen::Vector<Scalar, 2> rotcenter_c = midpoints.col(i) + vecs.col(i) * theta_cot_c;

        // Start and end x coordinate of the rectangle (Vertical segment)
        segments[j] = {rotcenter_c[1] - xi_c, rotcenter_c[1] + xi_c, rotcenter_c[0] - xi_c, weights[i]};
        segments[j + 1] = {rotcenter_c[1] - xi_c, rotcenter_c[1] + xi_c, rotcenter_c[0] + xi_c, -weights[i]};
    }

    lb = sweepLine2D(segments, segtree);

    // Generally unreachable but it might happen due to numerical error
    if (lb > ub) {
        ub = lb;
    }
}

template <typename Scalar> Scalar NodeStage2<Scalar>::volume() const { return theta[1] - theta[0]; }
/*****************NodeStage2 end******************/

} // namespace gmor
