/**
 * @file trde.cpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Re-implementation of TR-DE
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#include "trde.h"
#include "trde_node.hpp"

#include <omp.h>

#include <queue>

namespace trde {

TRDESolver::TRDESolver() : stage1_best{0.0f}, stage2_best{0.0f}, m_numInliers(0) {}

TRDESolver::~TRDESolver() = default;

Eigen::Matrix4f TRDESolver::solve(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                                  const std::vector<std::tuple<int, int, float>>& correspondences) {
    if (source.cols() < 3 || target.cols() < 3 || correspondences.size() < 3) {
        std::cerr << "Too few points for registration." << std::endl;
        return Eigen::Matrix4f::Identity();
    }
    // Extract correspondences
    Eigen::Matrix3Xf source_corrs(3, correspondences.size());
    Eigen::Matrix3Xf target_corrs(3, correspondences.size());
    for (size_t i = 0; i < correspondences.size(); ++i) {
        source_corrs.col(i) = source.col(std::get<0>(correspondences[i]));
        target_corrs.col(i) = target.col(std::get<1>(correspondences[i]));
    }

    // Generally necessary, since the BnB search space contains distance
    Eigen::Matrix4f sourceTransform, targetTransform;
    float scale = 1.0f;
    if (m_normalize) {
        scale = gmor::RegistrationBnBBase<float>::normalizePoints(source_corrs, target_corrs, sourceTransform,
                                                                  targetTransform);
    } else {
        sourceTransform.setIdentity();
        targetTransform.setIdentity();
        Eigen::Vector3f source_center = source_corrs.rowwise().mean();
        Eigen::Vector3f target_center = target_corrs.rowwise().mean();
        source_corrs.colwise() -= source_center;
        target_corrs.colwise() -= target_center;
        sourceTransform.topRightCorner<3, 1>() = -source_center;
        targetTransform.topRightCorner<3, 1>() = target_center;
    }
    m_noiseBound *= scale;

    stage1_2and1DOF(source_corrs, target_corrs, -M_PI, M_PI, 0.0, 1.25 * log(tan(9.0 * M_PI / 20.0)), -2.0, 2.0);
    filterInliersStage1(source_corrs, target_corrs);
    stage2_1and2DOF(source_corrs, target_corrs, -M_PI, M_PI, 0.0, 2.0);
    Eigen::Vector3f r_c = gmor::miller2RotAxis<float>(stage1_best[0], stage1_best[1]);
    Eigen::Vector3f e1, e2;
    gmor::genVertBaseVecs<float>(r_c, e1, e2);
    Eigen::Vector3f trans_mat = stage1_best[2] * r_c + stage2_best[1] * std::cos(stage2_best[0]) * e1 +
                                stage2_best[1] * std::sin(stage2_best[0]) * e2;
    Eigen::Matrix3f rot_mat = Eigen::AngleAxisf(stage2_best[2], r_c).toRotationMatrix();

    gmor::RegistrationBnBBase<float>::postRefinementSVD(source_corrs, target_corrs, rot_mat, trans_mat, 5,
                                                        m_noiseBound);

    Eigen::Matrix4f result_mat = Eigen::Matrix4f::Identity();
    result_mat.topLeftCorner<3, 3>() = rot_mat;
    result_mat.topRightCorner<3, 1>() = trans_mat;
    result_mat = targetTransform * result_mat * sourceTransform;
    m_noiseBound /= scale;
    return result_mat;
}

void TRDESolver::stage1_2and1DOF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target, float rx_l, float rx_r,
                                 float ry_l, float ry_r, float d_l, float d_r) {
    uint32_t best_lb = 0;

#pragma omp parallel for num_threads(m_numThreads) default(none) shared(source, target, best_lb)                       \
    firstprivate(rx_l, rx_r, ry_l, ry_r, d_l, d_r) schedule(dynamic, 1)
    for (int i = 0; i < 12; ++i) { // Search space is divided into 12 regions
        // No Eigen matrix in Node, so Eigen::aligned_allocator is not needed
        std::priority_queue<Node21Stage1<float>> queue_stage1;

        // Initialize the search space in a 2x2x3 grid
        const int rx_i = i / 6;
        const int ry_i = (i % 6) / 3;
        const int d_i = i % 3;
        const float rx_l_ = rx_l + rx_i * (rx_r - rx_l) * 0.5;
        const float rx_r_ = rx_l + (rx_i + 1) * (rx_r - rx_l) * 0.5;
        const float ry_l_ = ry_l + ry_i * (ry_r - ry_l) * 0.5;
        const float ry_r_ = ry_l + (ry_i + 1) * (ry_r - ry_l) * 0.5;
        const float d_l_ = d_l + d_i * (d_r - d_l) / 3.0;
        const float d_r_ = d_l + (d_i + 1) * (d_r - d_l) / 3.0;
        Node21Stage1<float> node_init(rx_l_, rx_r_, ry_l_, ry_r_, d_l_, d_r_);
        node_init.estULB(source, target, m_noiseBound);
        queue_stage1.push(node_init);

        while (!queue_stage1.empty()) {
            Node21Stage1<float> node = queue_stage1.top();
            queue_stage1.pop();

            // Prune current branch
            if (node.ub <= best_lb)
                continue;

#pragma omp critical
            {
                // Update best center and lower bound of inliers
                if (node.lb > best_lb) {
                    best_lb = node.lb;
                    stage1_best[0] = (node.rx[0] + node.rx[1]) / 2;
                    stage1_best[1] = (node.ry[0] + node.ry[1]) / 2;
                    stage1_best[2] = (node.d[0] + node.d[1]) / 2;
                }
            }

            // Stop splitting
            if (node.rx[1] - node.rx[0] < m_branch_eps && node.ry[1] - node.ry[0] < m_branch_eps &&
                node.d[1] - node.d[0] < m_branch_eps)
                continue;

            if (node.ub - node.lb < m_bound_eps)
                continue;

            // Split the widest dimension of current node
            auto nodes_split = node.split();
            for (auto& node_s : nodes_split) {
                node_s.estULB(source, target, m_noiseBound);
                queue_stage1.push(node_s);
            }
        }
    }
}

void TRDESolver::stage2_1and2DOF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target, float phi_l,
                                 float phi_r, float h_l, float h_r) {
    uint32_t best_lb = 0;
    Eigen::Vector3f r_c = gmor::miller2RotAxis<float>(stage1_best[0], stage1_best[1]);
    float d_c = stage1_best[2];

#pragma omp parallel for num_threads(m_numThreads) default(none) shared(source, target, best_lb)                       \
    firstprivate(phi_l, phi_r, h_l, h_r, r_c, d_c) schedule(dynamic, 1)
    for (int i = 0; i < 12; ++i) { // Outer BnB
        std::priority_queue<NodeOuterStage2<float>> queue_outer;

        // Initialize the search space in the polar coordinate system
        const int phi_i = i / 3;
        const int h_i = i % 3;
        const float phi_l_ = phi_l + phi_i * (phi_r - phi_l) / 4;
        const float phi_r_ = phi_l + (phi_i + 1) * (phi_r - phi_l) / 4;
        const float h_l_ = h_l + h_i * (h_r - h_l) / 3;
        const float h_r_ = h_l + (h_i + 1) * (h_r - h_l) / 3;

        NodeOuterStage2<float> node_init(phi_l_, phi_r_, h_l_, h_r_);
        innerBnB(source, target, r_c, d_c, node_init.phi, node_init.h, node_init.theta_best);
        node_init.estULB(source, target, m_noiseBound, r_c, d_c, m_numInliers);
        queue_outer.push(node_init);

        while (!queue_outer.empty()) {
            NodeOuterStage2<float> node = queue_outer.top();
            queue_outer.pop();

            // Prune current branch
            if (node.ub <= best_lb)
                continue;

#pragma omp critical
            {
                if (node.lb > best_lb) {
                    best_lb = node.lb;
                    stage2_best[0] = 0.5 * (node.phi[0] + node.phi[1]);
                    stage2_best[1] = 0.5 * (node.h[0] + node.h[1]);
                    stage2_best[2] = 0.5 * (node.theta_best[0] + node.theta_best[1]);
                }
            }

            // Stop splitting
            if (node.phi[1] - node.phi[0] < m_branch_eps && node.h[1] - node.h[0] < m_branch_eps)
                continue;

            if (node.ub - node.lb < m_bound_eps)
                continue;

            // Split the widest dimension of current node
            auto nodes_split = node.split();
            for (auto& node_s : nodes_split) {
                // First, estimate the best theta in the inner BnB (start with [-pi, pi])
                innerBnB(source, target, r_c, d_c, node_s.phi, node_s.h, node_s.theta_best);
                // Estimate the upper and lower bound with the best theta
                node_s.estULB(source, target, m_noiseBound, r_c, d_c, m_numInliers);
                queue_outer.push(node_s);
            }
        }
    }
}

void TRDESolver::innerBnB(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                          const Eigen::Vector<float, 3>& rc, const float& dc, const float* phi_lu, const float* h_lu,
                          float* theta_lu) {
    std::priority_queue<NodeInnerStage2<float>> queue_inner;
    theta_lu[0] = -M_PI;
    theta_lu[1] = M_PI;
    NodeInnerStage2<float> node_init(-M_PI, M_PI);
    node_init.estULB(source, target, m_noiseBound, rc, dc, phi_lu, h_lu, m_numInliers);
    queue_inner.push(node_init);
    uint32_t best_lb_inner = 0;
    while (!queue_inner.empty()) {
        NodeInnerStage2<float> node = queue_inner.top();
        queue_inner.pop();

        // Prune current branch
        if (node.ub <= best_lb_inner)
            continue;

        // Update best center and lower bound of inliers
        if (node.lb > best_lb_inner) {
            best_lb_inner = node.lb;
            theta_lu[0] = node.theta[0];
            theta_lu[1] = node.theta[1];
        }

        // Stop splitting
        if (node.theta[1] - node.theta[0] < m_branch_eps)
            continue;

        if (node.ub - node.lb < m_bound_eps)
            continue;

        // Split theta
        auto nodes_split = node.split();
        for (auto& node_s : nodes_split) {
            node_s.estULB(source, target, m_noiseBound, rc, dc, phi_lu, h_lu, m_numInliers);
            queue_inner.push(node_s);
        }
    }
}

void TRDESolver::filterInliersStage1(Eigen::Matrix3Xf& source, Eigen::Matrix3Xf& target) {
    Eigen::Vector3f r_c = gmor::miller2RotAxis<float>(stage1_best[0], stage1_best[1]);
    float d_c = stage1_best[2];

    m_numInliers = 0;
    for (int i = 0; i < source.cols(); ++i) {
        Eigen::Vector3f xy = source.col(i) - target.col(i);
        float dist_c = std::fabs(r_c.transpose() * xy + d_c);
        if (dist_c <= m_noiseBound) {
            source.col(m_numInliers) = source.col(i);
            target.col(m_numInliers) = target.col(i);
            m_numInliers++;
        }
    }
}

} // namespace trde
