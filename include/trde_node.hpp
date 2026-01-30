/**
 * @file trde_node.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Node of TR-DE
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include "registration_base.hpp"
#include "registration_utils.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace trde {

template <typename Scalar> class Node21Stage1 : public gmor::Node<uint32_t, Node21Stage1<Scalar>> {
  public:
    Scalar rx[2], ry[2], d[2]; // 2+1 DOF of rotation axis and displacement along it

    Node21Stage1(Scalar rx_l, Scalar rx_r, Scalar ry_l, Scalar ry_r, Scalar d_l, Scalar d_r);
    ~Node21Stage1() = default;

    std::vector<Node21Stage1<Scalar>> split();

    Scalar volume() const;

    // xi is the inlier threshold in Section 3.1
    void estULB(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& src,
                const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& tgt, const Scalar& xi);

    using gmor::Node<uint32_t, Node21Stage1<Scalar>>::lb;
    using gmor::Node<uint32_t, Node21Stage1<Scalar>>::ub;
    using Ptr = std::shared_ptr<Node21Stage1<Scalar>>;
    using Ptr_u = std::unique_ptr<Node21Stage1<Scalar>>;
};

template <typename Scalar> class NodeOuterStage2 : public gmor::Node<uint32_t, NodeOuterStage2<Scalar>> {
  public:
    Scalar phi[2], h[2];  // 2 DOF of translation vertical to the rotation axis
    Scalar theta_best[2]; // Rotation angle estimated in the inner BnB

    NodeOuterStage2(Scalar phi_l, Scalar phi_r, Scalar h_l, Scalar h_r);
    ~NodeOuterStage2() = default;

    std::vector<NodeOuterStage2<Scalar>> split();

    Scalar volume() const;

    void estULB(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& src,
                const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& tgt, const Scalar& xi,
                const Eigen::Vector<Scalar, 3>& rc, const Scalar& dc, const uint32_t& num_inliers);

    using gmor::Node<uint32_t, NodeOuterStage2<Scalar>>::lb;
    using gmor::Node<uint32_t, NodeOuterStage2<Scalar>>::ub;
    using Ptr = std::shared_ptr<NodeOuterStage2<Scalar>>;
    using Ptr_u = std::unique_ptr<NodeOuterStage2<Scalar>>;
};

template <typename Scalar> class NodeInnerStage2 : public gmor::Node<uint32_t, NodeInnerStage2<Scalar>> {
  public:
    Scalar theta[2]; // 1 DOF of rotation angle

    NodeInnerStage2(Scalar theta_l, Scalar theta_r);
    ~NodeInnerStage2() = default;

    std::vector<NodeInnerStage2<Scalar>> split();

    Scalar volume() const;

    void estULB(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& src,
                const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& tgt, Scalar xi, const Eigen::Vector<Scalar, 3>& rc,
                const Scalar& dc, const Scalar* phi_lu, const Scalar* h_lu, const uint32_t& num_inliers);

    using gmor::Node<uint32_t, NodeInnerStage2<Scalar>>::lb;
    using gmor::Node<uint32_t, NodeInnerStage2<Scalar>>::ub;
    using Ptr = std::shared_ptr<NodeInnerStage2<Scalar>>;
    using Ptr_u = std::unique_ptr<NodeInnerStage2<Scalar>>;
};

/*****************Implementation******************/

/*****************Node21Stage1******************/
template <typename Scalar>
Node21Stage1<Scalar>::Node21Stage1(Scalar rx_l, Scalar rx_r, Scalar ry_l, Scalar ry_r, Scalar d_l, Scalar d_r)
    : gmor::Node<uint32_t, Node21Stage1<Scalar>>(), rx{rx_l, rx_r}, ry{ry_l, ry_r}, d{d_l, d_r} {}

template <typename Scalar> std::vector<Node21Stage1<Scalar>> Node21Stage1<Scalar>::split() {
    std::vector<Node21Stage1<Scalar>> nodes;
    // Find dimension with the largest range and split into two nodes
    if (rx[1] - rx[0] > ry[1] - ry[0] && rx[1] - rx[0] > d[1] - d[0]) {
        nodes.emplace_back(rx[0], 0.5 * (rx[0] + rx[1]), ry[0], ry[1], d[0], d[1]);
        nodes.emplace_back(0.5 * (rx[0] + rx[1]), rx[1], ry[0], ry[1], d[0], d[1]);
    } else if (ry[1] - ry[0] > d[1] - d[0]) {
        nodes.emplace_back(rx[0], rx[1], ry[0], 0.5 * (ry[0] + ry[1]), d[0], d[1]);
        nodes.emplace_back(rx[0], rx[1], 0.5 * (ry[0] + ry[1]), ry[1], d[0], d[1]);
    } else {
        nodes.emplace_back(rx[0], rx[1], ry[0], ry[1], d[0], 0.5 * (d[0] + d[1]));
        nodes.emplace_back(rx[0], rx[1], ry[0], ry[1], 0.5 * (d[0] + d[1]), d[1]);
    }
    return nodes;
}

template <typename Scalar>
void Node21Stage1<Scalar>::estULB(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& src,
                                  const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& tgt, const Scalar& xi) {
    if (src.cols() < 3 || src.cols() != tgt.cols()) {
        std::cerr << "Input size error" << std::endl;
        return;
    }
    // r_c in Eq. (8b), and O in Fig. 3
    Eigen::Vector<Scalar, 3> r_c = gmor::miller2RotAxis<Scalar>(0.5 * (rx[0] + rx[1]), 0.5 * (ry[0] + ry[1]));
    // r_max is both A and B in Fig. 3. rx[0] can be replaced by rx[1], but ry[0] can't.
    Eigen::Vector<Scalar, 3> r_max = gmor::miller2RotAxis<Scalar>(rx[0], ry[0]);
    Eigen::Vector<Scalar, 3> r_diff = r_max - r_c;
    // tau and delta_d in Eq. (9)
    Scalar tau = r_diff.norm();
    Scalar delta_d = 0.5 * (d[1] - d[0]);

    // Upper bound in Eq. (9)
    uint32_t inliers_lb = 0;
    uint32_t inliers_ub = 0;
    for (int i = 0; i < src.cols(); ++i) {
        Eigen::Vector<Scalar, 3> xy = src.col(i) - tgt.col(i);
        Scalar dist_c = fabs(r_c.transpose() * xy + 0.5 * (d[0] + d[1]));
        if (dist_c <= xi) {
            inliers_lb++;
        }
        if (dist_c <= xi + tau * xy.norm() + delta_d) {
            inliers_ub++;
        }
    }
    lb = inliers_lb;
    ub = inliers_ub;
}

template <typename Scalar> Scalar Node21Stage1<Scalar>::volume() const {
    return (rx[1] - rx[0]) * (ry[1] - ry[0]) * (d[1] - d[0]);
}
/*****************Node21Stage1 end******************/

/*****************NodeOuterStage2******************/
template <typename Scalar>
NodeOuterStage2<Scalar>::NodeOuterStage2(Scalar phi_l, Scalar phi_r, Scalar h_l, Scalar h_r)
    : gmor::Node<uint32_t, NodeOuterStage2<Scalar>>(), phi{phi_l, phi_r}, h{h_l, h_r}, theta_best{-M_PI, M_PI} {}

template <typename Scalar> std::vector<NodeOuterStage2<Scalar>> NodeOuterStage2<Scalar>::split() {
    std::vector<NodeOuterStage2<Scalar>> nodes;
    // Find the dimension with the largest range and split into two nodes
    if (phi[1] - phi[0] > h[1] - h[0]) {
        nodes.emplace_back(phi[0], 0.5 * (phi[0] + phi[1]), h[0], h[1]);
        nodes.emplace_back(0.5 * (phi[0] + phi[1]), phi[1], h[0], h[1]);
    } else {
        nodes.emplace_back(phi[0], phi[1], h[0], 0.5 * (h[0] + h[1]));
        nodes.emplace_back(phi[0], phi[1], 0.5 * (h[0] + h[1]), h[1]);
    }
    return nodes;
}

template <typename Scalar>
void NodeOuterStage2<Scalar>::estULB(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& src,
                                     const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& tgt, const Scalar& xi,
                                     const Eigen::Vector<Scalar, 3>& rc, const Scalar& dc,
                                     const uint32_t& num_inliers) {
    if (src.cols() < 3 || src.cols() != tgt.cols()) {
        std::cerr << "Input size error" << std::endl;
        return;
    }
    // Arbitrary known orthogonal basis in Section 4.1
    Eigen::Vector<Scalar, 3> e1, e2;
    // Estimate $t_{\perp}$ using rotation axis in Eq. (7)
    gmor::genVertBaseVecs<Scalar>(rc, e1, e2);

    // eta_t in Eq. (11e)
    Scalar h_c = 0.5 * (h[0] + h[1]);
    Scalar phi_c = 0.5 * (phi[0] + phi[1]);
    Eigen::Vector<Scalar, 2> t_c_2d(h_c * std::cos(phi_c), h_c * std::sin(phi_c));
    Eigen::Vector<Scalar, 2> t_max_2d(h[1] * std::cos(phi[0]), h[1] * std::sin(phi[0]));
    Scalar eta_t = (t_c_2d - t_max_2d).norm();

    // Rotation matrix of axis rc and angle theta
    Scalar theta_c = 0.5 * (theta_best[0] + theta_best[1]);
    Eigen::Matrix<Scalar, 3, 3> rc_mat = Eigen::AngleAxis<Scalar>(theta_c, rc.normalized()).toRotationMatrix();

    // Translation vector of $ t_\parallel $ and $ t_\perp $
    Eigen::Vector<Scalar, 3> t_c = dc * rc + h_c * std::cos(phi_c) * e1 + h_c * std::sin(phi_c) * e2;

    // Upper bound in Eq. (9)
    uint32_t inliers_lb = 0;
    uint32_t inliers_ub = 0;
    for (uint32_t i = 0; i < num_inliers; ++i) {
        // Eq. (14)
        Eigen::Vector<Scalar, 3> xy = rc_mat * src.col(i) + t_c - tgt.col(i);
        Scalar dist_c = xy.norm();
        if (dist_c <= xi) {
            inliers_lb++;
        }
        if (dist_c <= xi + eta_t) {
            inliers_ub++;
        }
    }
    lb = inliers_lb;
    ub = inliers_ub;
}

template <typename Scalar> Scalar NodeOuterStage2<Scalar>::volume() const { return (phi[1] - phi[0]) * (h[1] - h[0]); }
/*****************NodeOuterStage2 end******************/

/*****************NodeInnerStage2******************/
template <typename Scalar>
NodeInnerStage2<Scalar>::NodeInnerStage2(Scalar theta_l, Scalar theta_r)
    : gmor::Node<uint32_t, NodeInnerStage2<Scalar>>(), theta{theta_l, theta_r} {}

template <typename Scalar> std::vector<NodeInnerStage2<Scalar>> NodeInnerStage2<Scalar>::split() {
    std::vector<NodeInnerStage2<Scalar>> nodes;
    nodes.emplace_back(theta[0], 0.5 * (theta[0] + theta[1]));
    nodes.emplace_back(0.5 * (theta[0] + theta[1]), theta[1]);
    return nodes;
}

template <typename Scalar>
void NodeInnerStage2<Scalar>::estULB(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& src,
                                     const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& tgt, Scalar xi,
                                     const Eigen::Vector<Scalar, 3>& rc, const Scalar& dc, const Scalar* phi_lu,
                                     const Scalar* h_lu, const uint32_t& num_inliers) {
    if (src.cols() < 3 || src.cols() != tgt.cols()) {
        std::cerr << "Input size error" << std::endl;
        return;
    }
    // Arbitrary known orthogonal basis in Section 4.1
    Eigen::Vector<Scalar, 3> e1, e2;
    // Estimate $t_{\perp}$ using rotation axis in Eq. (7)
    gmor::genVertBaseVecs(rc, e1, e2);

    // eta_t in Eq. (11e)
    Scalar h_c = 0.5 * (h_lu[0] + h_lu[1]);
    Scalar phi_c = 0.5 * (phi_lu[0] + phi_lu[1]);
    Eigen::Vector<Scalar, 2> t_c_2d(h_c * std::cos(phi_c), h_c * std::sin(phi_c));
    Eigen::Vector<Scalar, 2> t_max_2d(h_lu[1] * std::cos(phi_lu[0]), h_lu[1] * std::sin(phi_lu[0]));
    Scalar eta_t = (t_c_2d - t_max_2d).norm();

    // scale of eta_theta in Eq. (12)
    Scalar theta_scale = sqrt(2 * (1 - std::cos(0.5 * (theta[1] - theta[0]))));

    // Rotation matrix of axis rc and angle theta
    Scalar theta_c = 0.5 * (theta[0] + theta[1]);
    Eigen::Matrix<Scalar, 3, 3> rc_mat = Eigen::AngleAxis<Scalar>(theta_c, rc.normalized()).toRotationMatrix();

    // Translation vector of $ t_\parallel $ and $ t_\perp $
    Eigen::Vector<Scalar, 3> t_c = dc * rc + h_c * std::cos(phi_c) * e1 + h_c * std::sin(phi_c) * e2;

    // Upper bound in Eq. (9)
    uint32_t inliers_lb = 0;
    uint32_t inliers_ub = 0;
    for (uint32_t i = 0; i < num_inliers; ++i) {
        // Eq. (12)
        Eigen::Vector<Scalar, 3> x = src.col(i);
        Scalar eta_theta = theta_scale * x.norm();

        // Eq. (14)
        Eigen::Vector<Scalar, 3> xy = rc_mat * x + t_c - tgt.col(i);
        Scalar dist_c = xy.norm();
        if (dist_c <= xi) {
            inliers_lb++;
        }
        if (dist_c <= xi + eta_theta + eta_t) {
            inliers_ub++;
        }
    }
    lb = inliers_lb;
    ub = inliers_ub;
}

template <typename Scalar> Scalar NodeInnerStage2<Scalar>::volume() const { return theta[1] - theta[0]; }
/*****************NodeInnerStage2 end******************/

} // namespace trde
