/**
 * @file gmor.cpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Implementation of GMOR
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#include "gmor.h"
#include "gmor_node.hpp"
#include "registration_utils.hpp"

#include <omp.h>

#include <queue>

namespace gmor {

constexpr float CHI_1_95 = 1.960;
constexpr float CHI_2_95 = 2.448;
constexpr float CHI_2_95_SQUARE = 2.169;
constexpr float CHI_3_95 = 2.796;

// Use cube mapping
using ProjectionType = SphereProjCube<float>;
using Node2Stage1Type = Node2Stage1<float, ProjectionType>;

GMOSolver::GMOSolver() : stage2_best(0.0f), m_rho(0.25f), m_topk(12), m_trans_only(true), m_rot_near_z(false) {}

GMOSolver::~GMOSolver() = default;

void GMOSolver::setTopk(int top_k) { this->m_topk = top_k; }

void GMOSolver::setRho(float rho) { this->m_rho = rho; }

void GMOSolver::setRotNearZ(bool rot_near_z) { this->m_rot_near_z = rot_near_z; }

Eigen::Matrix4f GMOSolver::solve(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                                 const std::vector<std::tuple<int, int, float>>& correspondences) {
    if (source.cols() < 3 || target.cols() < 3 || correspondences.size() < 3) {
        std::cerr << "Too few points for registration." << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    std::vector<float> weights(correspondences.size());
    Eigen::Matrix3Xf source_corrs(3, correspondences.size());
    Eigen::Matrix3Xf target_corrs(3, correspondences.size());
    for (size_t i = 0; i < correspondences.size(); ++i) {
        source_corrs.col(i) = source.col(std::get<0>(correspondences[i]));
        target_corrs.col(i) = target.col(std::get<1>(correspondences[i]));
        weights[i] = std::get<2>(correspondences[i]);
    }

    Eigen::Matrix4f sourceTransform, targetTransform;
    float scale = 1.0f;
    if (m_normalize) {
        scale = gmor::RegistrationBnBBase<float>::normalizePoints(source_corrs, target_corrs, sourceTransform,
                                                                  targetTransform);
    } else {
        // Move to centroid
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

    // Stage1: Find the rotation axes on the hemisphere
    stage1_2DOF(source_corrs, target_corrs, weights, m_noiseBound * std::sqrt(2.0) / CHI_3_95 * CHI_1_95);

    // Select top-k axes and evaluate the score-based inliers in stage2
    std::vector<uint32_t> best_inliers;
    evalTopkAxes(source_corrs, target_corrs, weights, best_inliers);

    Eigen::Matrix3Xf source_filtered(3, best_inliers.size());
    Eigen::Matrix3Xf target_filtered(3, best_inliers.size());

    for (size_t j = 0; j < best_inliers.size(); ++j) {
        source_filtered.col(j) = source_corrs.col(best_inliers[j]);
        target_filtered.col(j) = target_corrs.col(best_inliers[j]);
    }

    // Initial guess with estimated inliers
    Eigen::Matrix4f result_mat = Eigen::umeyama(source_filtered, target_filtered, false);
    Eigen::Matrix3f rot_mat = result_mat.topLeftCorner<3, 3>();
    Eigen::Vector3f trans_mat = result_mat.topRightCorner<3, 1>();
    // Refinement with all the correspondences
    gmor::RegistrationBnBBase<float>::postRefinementTukey(source_corrs, target_corrs, rot_mat, trans_mat, 5,
                                                          m_noiseBound, 0.5);

    // Recover the scale
    result_mat.topLeftCorner<3, 3>() = rot_mat;
    result_mat.topRightCorner<3, 1>() = trans_mat;
    result_mat = targetTransform * result_mat * sourceTransform;
    m_noiseBound /= scale;
    return result_mat;
}

Eigen::Matrix3f GMOSolver::solve2D(const Eigen::Matrix2Xf& source, const Eigen::Matrix2Xf& target,
                                   const std::vector<std::tuple<int, int, float>>& correspondences) {
    if (source.cols() < 2 || target.cols() < 2 || correspondences.size() < 2) {
        std::cerr << "Too few points for registration." << std::endl;
        return Eigen::Matrix3f::Identity();
    }

    std::vector<float> weights(correspondences.size());
    Eigen::Matrix2Xf source_corrs(2, correspondences.size());
    Eigen::Matrix2Xf target_corrs(2, correspondences.size());
    for (size_t i = 0; i < correspondences.size(); ++i) {
        source_corrs.col(i) = source.col(std::get<0>(correspondences[i]));
        target_corrs.col(i) = target.col(std::get<1>(correspondences[i]));
        weights[i] = std::get<2>(correspondences[i]);
    }

    // Move to centroid
    Eigen::Matrix3f sourceTransform, targetTransform;
    {
        sourceTransform.setIdentity();
        targetTransform.setIdentity();
        Eigen::Vector2f source_center = source_corrs.rowwise().mean();
        Eigen::Vector2f target_center = target_corrs.rowwise().mean();
        source_corrs.colwise() -= source_center;
        target_corrs.colwise() -= target_center;
        sourceTransform.topRightCorner<2, 1>() = -source_center;
        targetTransform.topRightCorner<2, 1>() = target_center;
    }

    std::vector<uint32_t> indices_filtered;
    {
        Eigen::Matrix2Xf midpts = 0.5 * (source_corrs + target_corrs);
        Eigen::Matrix<float, 2, 2> rot_90;
        rot_90 << 0.0f, -1.0f, 1.0f, 0.0f;
        Eigen::Matrix2Xf vecs = 0.5 * rot_90 * (target_corrs - source_corrs);

        // Find the rotation angle, this noise bound is 2D
        float xi = m_noiseBound * std::sqrt(0.5) / CHI_2_95 * CHI_2_95_SQUARE;
        stage2_1DOF(midpts, vecs, weights, xi, 0.5 * m_branch_eps, 2 * M_PI - 0.5 * m_branch_eps);
        indices_filtered = filterInliersStage2(midpts, vecs, weights, xi);
    }

    if (indices_filtered.size() < 2) {
        std::cerr << "Too few points for registration." << std::endl;
        return Eigen::Matrix3f::Identity();
    }

    Eigen::Matrix2Xf source_filtered(2, indices_filtered.size());
    Eigen::Matrix2Xf target_filtered(2, indices_filtered.size());

    uint32_t j = 0;
    for (auto index : indices_filtered) {
        source_filtered.col(j) = source_corrs.col(index);
        target_filtered.col(j) = target_corrs.col(index);
        j++;
    }

    // Post 2D refinement
    Eigen::Matrix3f result_mat = Eigen::umeyama(source_filtered, target_filtered, false);
    Eigen::Matrix2f rot_mat = result_mat.topLeftCorner<2, 2>();
    Eigen::Vector2f trans_mat = result_mat.topRightCorner<2, 1>();
    gmor::RegistrationBnBBase<float>::postRefinementTukey(source_corrs, target_corrs, rot_mat, trans_mat, 5,
                                                          m_noiseBound, 0.5);
    result_mat.topLeftCorner<2, 2>() = rot_mat;
    result_mat.topRightCorner<2, 1>() = trans_mat;
    result_mat = targetTransform * result_mat * sourceTransform;

    return result_mat;
}

Eigen::Matrix4f GMOSolver::solvewithAxis(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                                         const std::vector<std::tuple<int, int, float>>& correspondences,
                                         const Eigen::Vector3f& axis) {
    if (source.cols() < 2 || target.cols() < 2 || correspondences.size() < 2) {
        std::cerr << "Too few points for registration." << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    std::vector<float> weights(correspondences.size());
    Eigen::Matrix3Xf source_corrs(3, correspondences.size());
    Eigen::Matrix3Xf target_corrs(3, correspondences.size());
    for (size_t i = 0; i < correspondences.size(); ++i) {
        source_corrs.col(i) = source.col(std::get<0>(correspondences[i]));
        target_corrs.col(i) = target.col(std::get<1>(correspondences[i]));
        weights[i] = std::get<2>(correspondences[i]);
    }

    Eigen::Matrix4f sourceTransform, targetTransform;
    float scale = 1.0f;
    if (m_normalize) {
        scale = gmor::RegistrationBnBBase<float>::normalizePoints(source_corrs, target_corrs, sourceTransform,
                                                                  targetTransform);
    } else {
        // Move to centroid
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

    Eigen::Vector<float, 3> e1, e2;
    gmor::genVertBaseVecs<float>(axis, e1, e2);
    Eigen::Matrix<float, 2, 3> e12;
    e12.row(0) = -e2.transpose();
    e12.row(1) = e1.transpose();

    e12.row(0) = e1.transpose();
    e12.row(1) = e2.transpose();

// #define EN_STRATEGY13
#ifdef EN_STRATEGY13
    // For comparison
    std::vector<uint32_t> best_inliers = strategy1_3DoF(source_corrs, target_corrs, weights, axis);
#else
    std::vector<uint32_t> best_inliers = strategy3_1DoF(source_corrs, target_corrs, weights, axis);
#endif

    // Filter inliers with the best score
    Eigen::Matrix3Xf source_filtered(3, best_inliers.size());
    Eigen::Matrix3Xf target_filtered(3, best_inliers.size());

    for (size_t j = 0; j < best_inliers.size(); ++j) {
        source_filtered.col(j) = source_corrs.col(best_inliers[j]);
        target_filtered.col(j) = target_corrs.col(best_inliers[j]);
    }

    float dist = ((target_filtered - source_filtered).transpose() * axis).mean();
    // Reuse
    Eigen::Matrix2Xf source_proj = e12 * source_filtered;
    Eigen::Matrix2Xf target_proj = e12 * target_filtered;
    Eigen::Matrix3f result_mat_2D = Eigen::umeyama(source_proj, target_proj, false);
    Eigen::Matrix2f rot_mat_2D = result_mat_2D.topLeftCorner<2, 2>();
    Eigen::Vector2f trans_mat_2D = result_mat_2D.topRightCorner<2, 1>();
    // Refinement with all the correspondences, need initial guess
    source_proj = e12 * source_corrs;
    target_proj = e12 * target_corrs;
    gmor::RegistrationBnBBase<float>::postRefinementTukey(source_proj, target_proj, rot_mat_2D, trans_mat_2D, 5,
                                                          m_noiseBound / CHI_3_95 * CHI_2_95, 0.5);
    float theta = std::atan2(rot_mat_2D(1, 0), rot_mat_2D(0, 0));
    Eigen::Matrix4f result_mat = Eigen::Matrix4f::Identity();
    result_mat.topLeftCorner<3, 3>() = Eigen::AngleAxisf(theta, axis).toRotationMatrix();
    result_mat.topRightCorner<3, 1>() = e12.transpose() * trans_mat_2D + dist * axis;
    result_mat = targetTransform * result_mat * sourceTransform;
    m_noiseBound /= scale;
    return result_mat;
}

void GMOSolver::stage1_2DOF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                            const std::vector<float>& weights, float xi) {
    // Early stop the unnecessary search
    float best_lb_topk = 0.0f;
    std::vector<float> queue_lb_vec(m_topk, 0);
    std::priority_queue<float, std::vector<float>, std::greater<>> queue_lb(queue_lb_vec.begin(), queue_lb_vec.end());

#pragma omp parallel num_threads(m_numThreads) default(none) firstprivate(xi)                                          \
    shared(source, target, weights, queue_lb, best_lb_topk)
    {
        std::array<Intervals<float, float>, 2> intervals;
        intervals[0].reserve(source.cols() << 1); // lb
        intervals[1].reserve(source.cols() << 1); // ub

#pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < 12; ++i) {
            if (m_rot_near_z && i < 8) {
                stage1_best[i][3] = 0;
                continue;
            }
            float best_lb = 0;

            std::priority_queue<Node2Stage1Type,
                                std::vector<Node2Stage1Type, Eigen::aligned_allocator<Node2Stage1Type>>>
                queue_stage1;

            // Search space is divided into 12 regions (depth = 1)
            Node2Stage1Type node_init(i, 1);
            node_init.estULB(source, target, weights, intervals, xi, best_lb, m_rho);
            queue_stage1.push(node_init);

            while (!queue_stage1.empty()) {
                Node2Stage1Type node = queue_stage1.top();
                queue_stage1.pop();

                // Prune current branch
                if (node.ub <= best_lb || node.ub <= best_lb_topk)
                    continue;

                // Update best center and lower bound of inliers
                if (node.lb > best_lb) {
                    best_lb = node.lb;
                    stage1_best[i].topLeftCorner<3, 1>() = node.getAxis();
                }

                // Stop splitting
                if ((node.region(0, 1) - node.region(0, 0)) < m_branch_eps &&
                    node.region(1, 1) - node.region(1, 0) < m_branch_eps)
                    continue;

                if (node.ub - node.lb < m_bound_eps)
                    continue;

                // Split the widest dimension of current node
                auto nodes_split = node.split();
                for (auto& node_s : nodes_split) {
                    node_s.estULB(source, target, weights, intervals, xi, best_lb, m_rho);
                    queue_stage1.push(node_s);
                }
            }
            stage1_best[i][3] = best_lb;
#pragma omp critical
            {
                queue_lb.push(best_lb);
                queue_lb.pop();
                best_lb_topk = queue_lb.top();
            }
        }
    }
}

float GMOSolver::evalTopkAxes(const Eigen::Matrix3Xf& source_corrs, const Eigen::Matrix3Xf& target_corrs,
                              const std::vector<float>& weights, std::vector<uint32_t>& best_inliers) {
    float xi = m_noiseBound * std::sqrt(0.5) / CHI_3_95 * CHI_2_95_SQUARE;
    float best_score = 0;
    // top-k rotation axes to reject duplicated axes
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> rc_set;
    // Sorted by lower bounds in stage1
    std::sort(stage1_best.begin(), stage1_best.end(), [](const auto& a, const auto& b) { return a[3] > b[3]; });
    for (int i = 0; i < m_topk; ++i) {
        // Sum weights of stage1 less than best stage2, stop iteration
        if (stage1_best[i][3] < best_score)
            break;

        Eigen::Vector3f rc(stage1_best[i].topLeftCorner<3, 1>());

        // Reject duplicated rc (Brute-force)
        bool duplicated = false;
        for (const auto& rc_i : rc_set) {
            float r_diff_cos = rc.dot(rc_i);
            // 1 - \delta^2 / 2, where \delta = sqrt(2) * m_branch_eps is diagnal length of the minimum square
            if (std::fabs(r_diff_cos) > 1 - m_branch_eps * m_branch_eps) {
                duplicated = true;
                break;
            }
        }
        if (duplicated)
            continue;

        // Filter inliers by rotation axis
        std::vector<float> weights_filtered;
        Eigen::Matrix2Xf midpts, vecs_proj;
        auto indices_stage1 =
            filterInliersProj(source_corrs, target_corrs, weights, midpts, vecs_proj, weights_filtered, rc,
                              m_noiseBound * std::sqrt(2.0) / CHI_3_95 * CHI_1_95);

        float score =
            stage2_1DOF(midpts, vecs_proj, weights_filtered, xi, 0.5 * m_branch_eps, 2 * M_PI - 0.5 * m_branch_eps);

        if (best_score < score) {
            auto indices_stage2 = filterInliersStage2(midpts, vecs_proj, weights_filtered, xi);
            best_score = score;
            best_inliers.clear();
            for (auto index : indices_stage2) {
                best_inliers.emplace_back(indices_stage1[index]);
            }
        }
        rc_set.push_back(rc);
    }
    return best_score;
}

float GMOSolver::stage2_1DOF(const Eigen::Matrix2Xf& midpts, const Eigen::Matrix2Xf& vecs,
                             const std::vector<float>& weights, float xi, float theta_l, float theta_r) {
    if (theta_l >= theta_r || theta_l < 0 || theta_r > 2 * M_PI) {
        std::cerr << "Invalid theta range" << std::endl;
        return 0.0f;
    }

    stage2_best = 0.0;
    m_trans_only = true;

    float best_lb = initBestLB(vecs, weights, xi);

#pragma omp parallel num_threads(m_numThreads) default(none) firstprivate(theta_l, theta_r, xi)                        \
    shared(best_lb, midpts, vecs, weights)
    {
        std::priority_queue<NodeStage2<float>> queue_theta;
        Segments<float> segments;
        segments.resize((midpts.cols() << 1) | 1);
        SegmentTreeZKW<float> segtree(midpts.cols() << 1);

#pragma omp for schedule(static, 1) nowait
        for (int i = 0; i < 12; ++i) {
            const float theta_l_ = theta_l + i * (theta_r - theta_l) / 12;
            const float theta_r_ = theta_l + (i + 1) * (theta_r - theta_l) / 12;
            NodeStage2<float> node_init(theta_l_, theta_r_);
            node_init.estUB(midpts, vecs, weights, segments, segtree, xi);
            queue_theta.push(node_init);
        }

        while (!queue_theta.empty()) {
            NodeStage2<float> node = queue_theta.top();
            queue_theta.pop();

            // Prune current branch
            if (node.ub <= best_lb)
                continue;

            // Lazy evaluation lb
            node.estLB(midpts, vecs, weights, segments, segtree, xi);

#pragma omp critical
            {
                if (node.lb > best_lb) {
                    best_lb = node.lb;
                    stage2_best = 0.5f * (node.theta[0] + node.theta[1]);
                    m_trans_only = false;
                }
            }

            // Stop splitting
            if ((node.theta[1] - node.theta[0]) < m_branch_eps)
                continue;

            if (node.ub - node.lb < m_bound_eps)
                continue;

            // Split the widest dimension of current node
            auto nodes_split = node.split();
            for (auto& node_s : nodes_split) {
                node_s.estUB(midpts, vecs, weights, segments, segtree, xi);
                queue_theta.push(node_s);
            }
        }
    }
    return best_lb;
}

std::vector<uint32_t> GMOSolver::strategy3_1DoF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                                                const std::vector<float>& weights, const Eigen::Vector3f& axis) {
    // Filter inliers by rotation angle on 2D plane
    Eigen::Matrix3Xf add_mat = target + source;
    Eigen::Matrix3Xf sub_mat = target - source;
    Eigen::Matrix2Xf midpts, vecs_proj;
    Eigen::Vector<float, 3> e1, e2;
    gmor::genVertBaseVecs<float>(axis, e1, e2);
    Eigen::Matrix<float, 2, 3> e12;
    e12.row(0) = -e2.transpose();
    e12.row(1) = e1.transpose();
    vecs_proj = 0.5 * e12 * sub_mat;

    e12.row(0) = e1.transpose();
    e12.row(1) = e2.transpose();
    midpts = 0.5 * e12 * add_mat;

    float xi = m_noiseBound * std::sqrt(0.5) / CHI_3_95 * CHI_2_95_SQUARE;
    stage2_1DOF(midpts, vecs_proj, weights, xi, 0.5 * m_branch_eps, 2 * M_PI - 0.5 * m_branch_eps);

    auto indices_stage2 = filterInliersStage2(midpts, vecs_proj, weights, xi);

    CenterIndices<float, float> centers;
    centers.reserve(indices_stage2.size());
    for (auto i : indices_stage2) {
        // Projection onto r_c
        float d_proj = (target.col(i) - source.col(i)).dot(axis);
        centers.emplace_back(d_proj, weights[i], i);
    }
    auto indices_stage1 =
        intervalStabbingFilterIndices<float, float>(centers, 2 * m_noiseBound * std::sqrt(2.0) / CHI_3_95 * CHI_1_95);
    size_t numInliers = indices_stage1.size();
    
    std::vector<float> weights_filtered;
    weights_filtered.resize(numInliers);

    for (size_t j = 0; j < indices_stage1.size(); ++j) {
        add_mat.col(j) = target.col(indices_stage1[j]) + source.col(indices_stage1[j]);
        sub_mat.col(j) = target.col(indices_stage1[j]) - source.col(indices_stage1[j]);
        weights_filtered[j] = weights[indices_stage1[j]];
    }

    std::vector<uint32_t> best_inliers;
    best_inliers.reserve(indices_stage1.size());
    for (auto index : indices_stage1) {
        best_inliers.emplace_back(index);
    }
    return best_inliers;
}

std::vector<uint32_t> GMOSolver::strategy1_3DoF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                                                const std::vector<float>& weights, const Eigen::Vector3f& axis) {
    // Filter inliers by rotation axis
    std::vector<float> weights_filtered;
    Eigen::Matrix2Xf midpts, vecs_proj;
    CenterIndices<float, float> centers;
    centers.reserve(source.cols());
    for (int i = 0; i < source.cols(); ++i) {
        // Projection onto r_c
        float d_proj = (target.col(i) - source.col(i)).dot(axis);
        centers.emplace_back(d_proj, weights[i], i);
    }
    auto indices_stage1 =
        intervalStabbingFilterIndices<float, float>(centers, 2 * m_noiseBound * std::sqrt(2.0) / CHI_3_95 * CHI_1_95);
    size_t numInliers = indices_stage1.size();
    Eigen::Matrix3Xf add_mat(3, numInliers);
    Eigen::Matrix3Xf sub_mat(3, numInliers);
    weights_filtered.resize(numInliers);

    for (size_t j = 0; j < indices_stage1.size(); ++j) {
        add_mat.col(j) = target.col(indices_stage1[j]) + source.col(indices_stage1[j]);
        sub_mat.col(j) = target.col(indices_stage1[j]) - source.col(indices_stage1[j]);
        weights_filtered[j] = weights[indices_stage1[j]];
    }

    Eigen::Vector<float, 3> e1, e2;
    gmor::genVertBaseVecs<float>(axis, e1, e2);
    Eigen::Matrix<float, 2, 3> e12;
    e12.row(0) = -e2.transpose();
    e12.row(1) = e1.transpose();
    vecs_proj = 0.5 * e12 * sub_mat;

    e12.row(0) = e1.transpose();
    e12.row(1) = e2.transpose();
    midpts = 0.5 * e12 * add_mat;

    // Stage2: Find the rotation angle
    float xi = m_noiseBound * std::sqrt(0.5) / CHI_3_95 * CHI_2_95_SQUARE;
    stage2_1DOF(midpts, vecs_proj, weights_filtered, xi, 0.5 * m_branch_eps, 2 * M_PI - 0.5 * m_branch_eps);

    auto indices_stage2 = filterInliersStage2(midpts, vecs_proj, weights_filtered, xi);
    std::vector<uint32_t> best_inliers;
    best_inliers.reserve(indices_stage2.size());
    for (auto index : indices_stage2) {
        best_inliers.emplace_back(indices_stage1[index]);
    }
    return best_inliers;
}

float GMOSolver::initBestLB(const Eigen::Matrix2Xf& vecs, const std::vector<float>& weights, float xi) {
    // Initialize segments
    uint32_t num_segs = vecs.cols() << 1;
    SegmentTreeZKW<float> segtree(num_segs);
    Segments<float> segs;
    segs.reserve(num_segs + 1);
    segs.emplace_back(0, 0, 0, 0);

    for (int i = 0; i < vecs.cols(); ++i) {
        Eigen::Vector<float, 2> vec = vecs.col(i);
        // Start x coordinate of the rectangle (Vertical segment)
        segs.emplace_back(vec[1] - xi, vec[1] + xi, vec[0] - xi, weights[i]);
        // End x coordinate of the rectangle (Vertical segment)
        segs.emplace_back(vec[1] - xi, vec[1] + xi, vec[0] + xi, -weights[i]);
    }

    return sweepLine2D(segs, segtree);
}
std::vector<uint32_t> GMOSolver::filterInliersProj(const Eigen::Matrix3Xf& src, const Eigen::Matrix3Xf& tgt,
                                                   const std::vector<float>& weights, Eigen::Matrix2Xf& midpts,
                                                   Eigen::Matrix2Xf& vecs_proj, std::vector<float>& weights_filtered,
                                                   Eigen::Vector3f& rc, float xi) {
    CenterIndices<float, float> centers;
    centers.reserve(src.cols());
    for (int i = 0; i < src.cols(); ++i) {
        float d_proj = (tgt.col(i) - src.col(i)).dot(rc);
        centers.emplace_back(d_proj, weights[i], i);
    }
    auto indices = intervalStabbingFilterIndices<float, float>(centers, 2 * xi);
    size_t numInliers = indices.size();
    Eigen::Matrix3Xf add_mat(3, numInliers);
    Eigen::Matrix3Xf sub_mat(3, numInliers);
    weights_filtered.resize(numInliers);

    for (size_t j = 0; j < indices.size(); ++j) {
        add_mat.col(j) = tgt.col(indices[j]) + src.col(indices[j]);
        sub_mat.col(j) = tgt.col(indices[j]) - src.col(indices[j]);
        weights_filtered[j] = weights[indices[j]];
    }

    Eigen::Vector3f centroid;
    gmor::planeFitting(sub_mat, rc, centroid);

    Eigen::Vector<float, 3> e1, e2;
    gmor::genVertBaseVecs<float>(rc, e1, e2);
    Eigen::Matrix<float, 2, 3> e12;
    e12.row(0) = e1.transpose();
    e12.row(1) = e2.transpose();

    midpts = 0.5 * e12 * add_mat;

    e12.row(0) = -e2.transpose();
    e12.row(1) = e1.transpose();
    vecs_proj = 0.5 * e12 * sub_mat;

    return indices;
}

std::vector<uint32_t> GMOSolver::filterInliersStage2(const Eigen::Matrix2Xf& midpts, const Eigen::Matrix2Xf& vecs,
                                                     const std::vector<float>& weights, float xi) const {
    return m_trans_only ? filterInliersTrans(vecs, weights, xi) : filterInliersRot(midpts, vecs, weights, xi);
}

std::vector<uint32_t> GMOSolver::filterInliersRot(const Eigen::Matrix2Xf& midpts, const Eigen::Matrix2Xf& vecs,
                                                  const std::vector<float>& weights, float xi) const {
    float theta_cot_c = std::tan(M_PI_2 - 0.5 * stage2_best); // cot(theta/2)
    float xi_c = xi * std::sqrt(1 + theta_cot_c * theta_cot_c);

    uint32_t num_segs = midpts.cols() << 1;
    SegmentTreeZKW<float> segtree(num_segs);
    Segments<float> segs;
    segs.reserve(num_segs + 1);
    segs.emplace_back(0, 0, 0, 0);

    for (int i = 0; i < midpts.cols(); ++i) {
        Eigen::Vector<float, 2> rotcenter_c = midpts.col(i) + vecs.col(i) * theta_cot_c;
        segs.emplace_back(rotcenter_c[1] - xi_c, rotcenter_c[1] + xi_c, rotcenter_c[0] - xi_c, weights[i]);
        segs.emplace_back(rotcenter_c[1] - xi_c, rotcenter_c[1] + xi_c, rotcenter_c[0] + xi_c, -weights[i]);
    }

    // Indices of segments from 1 to num_segs
    std::vector<uint32_t> indices(num_segs + 1);
    std::iota(indices.begin() + 1, indices.end(), 1);
    // Record the index of the sorted segments by y in original order
    std::sort(indices.begin() + 1, indices.end(), [&](const auto& i, const auto& j) {
        return (i & 1 ? segs[i].yl : segs[i].yr) < (j & 1 ? segs[j].yl : segs[j].yr);
    });

    for (uint32_t i = 1; i <= num_segs; ++i) {
        if (indices[i] & 1) {
            segs[indices[i]].yl_i = segs[indices[i] + 1].yl_i = i;
        } else {
            segs[indices[i]].yr_i = segs[indices[i] - 1].yr_i = i;
        }
    }

    std::iota(indices.begin() + 1, indices.end(), 1);
    std::sort(indices.begin() + 1, indices.end(), [&](const auto& i, const auto& j) { return segs[i] < segs[j]; });

    segtree.build(num_segs);
    float max_val = 0;
    uint32_t max_idx = 0;
    for (uint32_t i = 1; i <= num_segs; ++i) {
        segtree.update(segs[indices[i]].yl_i, segs[indices[i]].yr_i, segs[indices[i]].weight);
        if (max_val < segtree.getMax()) {
            max_val = segtree.getMax();
            max_idx = i;
        }
    }

    std::unordered_set<uint32_t> indices_filter;
    for (uint32_t i = 1; i <= max_idx; ++i) {
        if (segs[indices[i]].weight > 0) {
            indices_filter.insert((indices[i] - 1) >> 1);
        } else {
            indices_filter.erase((indices[i] - 1) >> 1);
        }
    }

    CenterIndices<float, float> centers;
    centers.reserve(indices_filter.size());
    for (auto index : indices_filter) {
        Eigen::Vector<float, 2> rotcenter_c = midpts.col(index) + vecs.col(index) * theta_cot_c;
        centers.emplace_back(rotcenter_c[1], weights[index], index);
    }

    auto indices_out = intervalStabbingFilterIndices<float>(centers, 2 * xi_c);

    return indices_out;
}
std::vector<uint32_t> GMOSolver::filterInliersTrans(const Eigen::Matrix2Xf& vecs, const std::vector<float>& weights,
                                                    float xi) const {
    uint32_t num_segs = vecs.cols() << 1;
    SegmentTreeZKW<float> segtree(num_segs);
    Segments<float> segs;
    segs.reserve(num_segs + 1);
    segs.emplace_back(0, 0, 0, 0);

    for (int i = 0; i < vecs.cols(); ++i) {
        Eigen::Vector<float, 2> vec = vecs.col(i);
        segs.emplace_back(vec[1] - xi, vec[1] + xi, vec[0] - xi, weights[i]);
        segs.emplace_back(vec[1] - xi, vec[1] + xi, vec[0] + xi, -weights[i]);
    }

    // Indices of segments from 1 to num_segs
    std::vector<uint32_t> indices(num_segs + 1);
    std::iota(indices.begin() + 1, indices.end(), 1);
    // Record the index of the sorted segments by y in original order
    std::sort(indices.begin() + 1, indices.end(), [&](const auto& i, const auto& j) {
        return (i & 1 ? segs[i].yl : segs[i].yr) < (j & 1 ? segs[j].yl : segs[j].yr);
    });

    for (uint32_t i = 1; i <= num_segs; ++i) {
        if (indices[i] & 1) {
            segs[indices[i]].yl_i = segs[indices[i] + 1].yl_i = i;
        } else {
            segs[indices[i]].yr_i = segs[indices[i] - 1].yr_i = i;
        }
    }

    std::iota(indices.begin() + 1, indices.end(), 1);
    std::sort(indices.begin() + 1, indices.end(), [&](const auto& i, const auto& j) { return segs[i] < segs[j]; });

    segtree.build(num_segs);
    float max_val = 0;
    uint32_t max_idx = 0;
    for (uint32_t i = 1; i <= num_segs; ++i) {
        segtree.update(segs[indices[i]].yl_i, segs[indices[i]].yr_i, segs[indices[i]].weight);
        if (max_val < segtree.getMax()) {
            max_val = segtree.getMax();
            max_idx = i;
        }
    }

    std::unordered_set<uint32_t> indices_filter;
    for (uint32_t i = 1; i <= max_idx; ++i) {
        if (segs[indices[i]].weight > 0) {
            indices_filter.insert((indices[i] - 1) >> 1);
        } else {
            indices_filter.erase((indices[i] - 1) >> 1);
        }
    }

    CenterIndices<float, float> centers;
    centers.reserve(indices_filter.size());
    for (auto index : indices_filter) {
        centers.emplace_back(vecs.col(index)[1], weights[index], index);
    }

    auto indices_out = intervalStabbingFilterIndices<float>(centers, 2 * xi);

    return indices_out;
}

} // namespace gmor
