/**
 * @file registration_base.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Base class for registration using branch-and-bound (BnB)
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace gmor {

/**
 * @brief Base node class for best-first search in std::priority_queue.
 *
 * @tparam T Type of lower and upper bound.
 * @tparam Derived Type of derived node class.
 */
template <typename T, typename Derived> class Node {
  public:
    // lower and upper bound of inliers
    T lb, ub;

    // For default std::less<> in priority_queue (maximize upper bound)
    bool operator<(const Derived& other) const {
        return ub < other.ub || (!(other.ub < ub) && static_cast<const Derived*>(this)->volume() < other.volume());
    }

  protected:
    Node() = default;
    ~Node() = default;
};

template <typename Scalar> class RegistrationBnBBase {
  public:
    RegistrationBnBBase();
    virtual ~RegistrationBnBBase();

    void setNoiseBound(Scalar xi);

    void setNormalize(bool normalize);

    void setNumThreads(int numThreads);

    void setBranchThreshold(Scalar eps);

    void setBoundThreshold(Scalar eps);

    /**
     * @brief Branch-and-bound registration of point clouds with correspondences
     *
     * @param source Source points
     * @param target Target points
     * @param correspondences Correspondences and weights from source to target points
     * @return Eigen::Matrix<Scalar, 4, 4> Homogeneous rigid transformation matrix
     *
     * For compatibility with pcl correspondences, the types in tuple are the same as index_query (int), index_match
     * (int), weight (float).
     */
    virtual Eigen::Matrix<Scalar, 4, 4> solve(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& source,
                                              const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& target,
                                              const std::vector<std::tuple<int, int, float>>& correspondences) = 0;

    /**
     * @brief Normalize source and target point clouds to [-1, 1]
     *
     * @param source Input source point cloud, output scaled cloud
     * @param target Input target point cloud, output scaled cloud
     * @param source_transform Output transformation of source to scaled source
     * @param target_inv_transform Output transformation of scaled target to target
     * @return Scalar Scale factor
     */
    static Scalar normalizePoints(Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& source,
                                  Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& target,
                                  Eigen::Matrix<Scalar, 4, 4>& source_transform,
                                  Eigen::Matrix<Scalar, 4, 4>& target_inv_transform);

    /**
     * @brief Iterations using inliers within noisebound. Need initial guess of rotation and translation.
     *
     * @tparam Dim
     * @param source Source points
     * @param target Target points
     * @param rotation Initial guess of rotation
     * @param translation Initial guess of translation
     * @param num_iter Number of iterations
     * @param xi Threshold of prior noise
     *
     * p.s. Generally for engineering applications, the transformation should be further estimated by fine registration
     * algorithm (e.g. ICP) using the entire point cloud and the initial guess of rotation and translation.
     */
    template <int Dim>
    static void postRefinementSVD(const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& source,
                                  const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& target,
                                  Eigen::Matrix<Scalar, Dim, Dim>& rotation, Eigen::Vector<Scalar, Dim>& translation,
                                  int num_iter, Scalar xi);
    template <int Dim>
    static void postRefinementTukey(const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& source,
                                    const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& target,
                                    Eigen::Matrix<Scalar, Dim, Dim>& rotation, Eigen::Vector<Scalar, Dim>& translation,
                                    int num_iter, Scalar xi, Scalar lambda, Scalar tuning = 4.685);
    template <int Dim>
    static void weightedSVD(const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& source,
                            const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& target,
                            const std::vector<Scalar>& weights, Eigen::Matrix<Scalar, Dim, Dim>& rotation,
                            Eigen::Vector<Scalar, Dim>& translation);

    using Ptr = std::shared_ptr<RegistrationBnBBase>;
    using Ptr_u = std::unique_ptr<RegistrationBnBBase>;

  protected:
    // Threshold of prior noise, search space length, and bound length
    Scalar m_noiseBound, m_branch_eps, m_bound_eps;
    int m_numThreads; // Number of threads (Recommended 12)
    bool m_normalize; // Normalize the input points to [-1, 1]
};

/*****************Implementation******************/

/*****************RegistrationBase******************/
template <typename Scalar>
RegistrationBnBBase<Scalar>::RegistrationBnBBase()
    : m_noiseBound(0.10), m_branch_eps(0.05), m_bound_eps(1e-3), m_numThreads(12), m_normalize(false) {}

template <typename Scalar> RegistrationBnBBase<Scalar>::~RegistrationBnBBase() = default;

template <typename Scalar> void RegistrationBnBBase<Scalar>::setNoiseBound(Scalar xi) { m_noiseBound = xi; }

template <typename Scalar> void RegistrationBnBBase<Scalar>::setNormalize(bool normalize) { m_normalize = normalize; }

template <typename Scalar> void RegistrationBnBBase<Scalar>::setNumThreads(int numThreads) {
    m_numThreads = numThreads > 0 ? numThreads : 1;
}

template <typename Scalar> void RegistrationBnBBase<Scalar>::setBranchThreshold(Scalar eps) { m_branch_eps = eps; }

template <typename Scalar> void RegistrationBnBBase<Scalar>::setBoundThreshold(Scalar eps) { m_bound_eps = eps; }

template <typename Scalar>
Scalar RegistrationBnBBase<Scalar>::normalizePoints(Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& source,
                                                    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& target,
                                                    Eigen::Matrix<Scalar, 4, 4>& source_transform,
                                                    Eigen::Matrix<Scalar, 4, 4>& target_inv_transform) {
    // Find centers and bounding boxes
    Eigen::Vector<Scalar, 3> source_min_pt, source_max_pt, target_min_pt, target_max_pt;
    source_min_pt = source.rowwise().minCoeff();
    source_max_pt = source.rowwise().maxCoeff();
    target_min_pt = target.rowwise().minCoeff();
    target_max_pt = target.rowwise().maxCoeff();
    Eigen::Vector<Scalar, 3> source_center, target_center;
    source_center = source.rowwise().mean();
    target_center = target.rowwise().mean();

    // Max half length of bounding boxes
    Scalar max_length = 1e-3;
    for (int i = 0; i < 3; ++i) {
        Scalar src_len = std::max(source_max_pt[i] - source_center[i], source_center[i] - source_min_pt[i]);
        Scalar tgt_len = std::max(target_max_pt[i] - target_center[i], target_center[i] - target_min_pt[i]);
        max_length = std::max(max_length, std::max(src_len, tgt_len));
    }

    // Scaling source and target clouds to [-1, 1] cube
    Scalar scale = 1.0 / max_length;
    source.colwise() -= source_center;
    target.colwise() -= target_center;
    source *= scale;
    target *= scale;

    source_transform.setIdentity();
    target_inv_transform.setIdentity();
    source_transform.template topLeftCorner<3, 3>() = Eigen::DiagonalMatrix<Scalar, 3>(scale, scale, scale);
    source_transform.template topRightCorner<3, 1>() = -source_center * scale;
    target_inv_transform.template topLeftCorner<3, 3>() =
        Eigen::DiagonalMatrix<Scalar, 3>(max_length, max_length, max_length);
    target_inv_transform.template topRightCorner<3, 1>() = target_center;

    return scale;
}

template <typename Scalar>
template <int Dim>
void RegistrationBnBBase<Scalar>::postRefinementSVD(const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& source,
                                                    const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& target,
                                                    Eigen::Matrix<Scalar, Dim, Dim>& rotation,
                                                    Eigen::Vector<Scalar, Dim>& translation, int num_iter, Scalar xi) {
    // Num iterations
    for (int i = 0; i < num_iter; ++i) {
        int inliers_num = 0;
        std::vector<int> inliers_index;

        // Estimate inliers num
        for (int j = 0; j < source.cols(); ++j) {
            Scalar error = (target.col(j) - rotation * source.col(j) - translation).norm();
            if (error <= xi) {
                inliers_index.push_back(j);
                inliers_num++;
            }
        }

        if (inliers_num < Dim) {
            std::cerr << "Not enough inliers for post refinement, use initial guess" << std::endl;
            return;
        }

        Eigen::Matrix<Scalar, Dim, Eigen::Dynamic> inliers_src(Dim, inliers_num);
        Eigen::Matrix<Scalar, Dim, Eigen::Dynamic> inliers_tgt(Dim, inliers_num);

        // Filter inliers
        for (int j = 0; j < inliers_num; j++) {
            inliers_src.col(j) = source.col(inliers_index[j]);
            inliers_tgt.col(j) = target.col(inliers_index[j]);
        }

        // Estimate rigid transformation matrix
        Eigen::Matrix<Scalar, Dim + 1, Dim + 1> trans_mat = Eigen::umeyama(inliers_src, inliers_tgt, false);
        rotation = trans_mat.template topLeftCorner<Dim, Dim>();
        translation = trans_mat.template topRightCorner<Dim, 1>();
    }
}

template <typename Scalar>
template <int Dim>
void RegistrationBnBBase<Scalar>::postRefinementTukey(const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& source,
                                                      const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& target,
                                                      Eigen::Matrix<Scalar, Dim, Dim>& rotation,
                                                      Eigen::Vector<Scalar, Dim>& translation, int num_iter, Scalar xi,
                                                      Scalar lambda, Scalar tuning) {
    // Num iterations
    for (int i = 0; i < num_iter; ++i) {
        std::vector<int> inliers_index;
        std::vector<Scalar> inliers_weights, inliers_residuals;

        // Estimate inliers num
        for (int j = 0; j < source.cols(); ++j) {
            Scalar error = (target.col(j) - rotation * source.col(j) - translation).norm();
            if (error <= xi) {
                inliers_index.push_back(j);
                inliers_residuals.push_back(error);
            }
        }

        if (inliers_index.size() < Dim) {
            std::cerr << "Not enough inliers for post refinement, use initial guess" << std::endl;
            return;
        }

        // Copy residuals
        std::vector<Scalar> inliers_residuals_sorted = inliers_residuals;
        size_t n = inliers_residuals_sorted.size();
        size_t mid = n >> 1;
        // MAD (median absolute deviation)
        Scalar median;
        std::nth_element(inliers_residuals_sorted.begin(), inliers_residuals_sorted.begin() + mid,
                         inliers_residuals_sorted.end());
        if (n & 1) {
            median = inliers_residuals_sorted[mid];
        } else {
            median =
                0.5 * (*std::max_element(inliers_residuals_sorted.begin(), inliers_residuals_sorted.begin() + mid) +
                       inliers_residuals_sorted[mid]);
        }
        // Tuning is 4.685 by default
        Scalar c = tuning * median;
        if (c < 1e-6)
            c = xi;

        Eigen::Matrix<Scalar, Dim, Eigen::Dynamic> inliers_src(Dim, inliers_index.size());
        Eigen::Matrix<Scalar, Dim, Eigen::Dynamic> inliers_tgt(Dim, inliers_index.size());
        inliers_weights.reserve(inliers_index.size());

        for (size_t j = 0; j < inliers_index.size(); j++) {
            inliers_src.col(j) = source.col(inliers_index[j]);
            inliers_tgt.col(j) = target.col(inliers_index[j]);
            // Tukey biweight
            inliers_weights.push_back(
                inliers_residuals[j] < c ? std::pow((1 - std::pow(inliers_residuals[j] / c, 2)), 2) : 0);
        }

        weightedSVD(inliers_src, inliers_tgt, inliers_weights, rotation, translation);

        xi = median + (xi - median) * lambda;
    }
}

template <typename Scalar>
template <int Dim>
void RegistrationBnBBase<Scalar>::weightedSVD(const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& source,
                                              const Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>& target,
                                              const std::vector<Scalar>& weights,
                                              Eigen::Matrix<Scalar, Dim, Dim>& rotation,
                                              Eigen::Vector<Scalar, Dim>& translation) {
    if (source.cols() != target.cols() || source.cols() != (int)weights.size()) {
        std::cerr << "Size of source, target and weights are not equal" << std::endl;
        return;
    }
    // Filter inliers and weighted centroid
    Scalar weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    // If weight_sum is too small, use non-weighted SVD
    if (weight_sum < 1e-6) {
        Eigen::Matrix<Scalar, Dim + 1, Dim + 1> trans_mat = Eigen::umeyama(source, target, false);
        rotation = trans_mat.template topLeftCorner<Dim, Dim>();
        translation = trans_mat.template topRightCorner<Dim, 1>();
        return;
    }

    Eigen::Map<const Eigen::Vector<Scalar, Eigen::Dynamic>> weights_mat(weights.data(), weights.size());
    Eigen::Vector<Scalar, Dim> centroid_src = source * weights_mat / weight_sum;
    Eigen::Vector<Scalar, Dim> centroid_tgt = target * weights_mat / weight_sum;

    Eigen::Matrix<Scalar, Dim, Dim> corr_mat =
        (source.colwise() - centroid_src) * weights_mat.asDiagonal() * (target.colwise() - centroid_tgt).transpose();
    Eigen::JacobiSVD<Eigen::Matrix<Scalar, Dim, Dim>> svd(corr_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    rotation = svd.matrixV() * svd.matrixU().transpose();
    if (rotation.determinant() < 0) {
        // Sigular values arranged from large to small
        Eigen::Vector<Scalar, Dim> w_diag = Eigen::Vector<Scalar, Dim>::Ones();
        w_diag(Dim - 1) = -1;
        rotation = svd.matrixV() * Eigen::DiagonalMatrix<Scalar, Dim>(w_diag) * svd.matrixU().transpose();
    }
    translation = centroid_tgt - rotation * centroid_src;
}

/*****************RegistrationBase end******************/

} // namespace gmor
