/**
 * @file gmor.h
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Registration based on geometric maxmum overlapping
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include "registration_base.hpp"

namespace gmor {

class GMOSolver : public gmor::RegistrationBnBBase<float> {
  public:
    GMOSolver();
    ~GMOSolver() override;

    void setTopk(int top_k);
    void setRho(float rho);
    void setRotNearZ(bool rot_near_z);

    // General 3D registration
    Eigen::Matrix4f solve(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                          const std::vector<std::tuple<int, int, float>>& correspondences) override;

    /**
     * @brief Public interface of the 2D rigid registration.
     *
     * @param source Source 2D points
     * @param target Target 2D points
     * @param correspondences Weighted correspondences from source to target points
     * @return Eigen::Matrix3f 2D rigid transformation
     */
    Eigen::Matrix3f solve2D(const Eigen::Matrix2Xf& source, const Eigen::Matrix2Xf& target,
                            const std::vector<std::tuple<int, int, float>>& correspondences);

    /**
     * @brief 4DOF registration with known rotation axis.
     *
     * @param source Source 3D points
     * @param target Target 3D points
     * @param correspondences Weighted correspondences from source to target points
     * @param axis Rotation axis
     * @return Eigen::Matrix4f 3D rigid transformation
     */
    Eigen::Matrix4f solvewithAxis(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                                  const std::vector<std::tuple<int, int, float>>& correspondences,
                                  const Eigen::Vector3f& axis);

    using Ptr = std::shared_ptr<GMOSolver>;
    using Ptr_u = std::unique_ptr<GMOSolver>;

  protected:
    // Stage I: Searching 2 DOF rotation axis
    void stage1_2DOF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target, const std::vector<float>& weights,
                     float xi);
    // Stage II: Top-k selection and searching 1 DOF of angle
    float evalTopkAxes(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                       const std::vector<float>& weights, std::vector<uint32_t>& best_inliers);
    float stage2_1DOF(const Eigen::Matrix2Xf& midpts, const Eigen::Matrix2Xf& vecs, const std::vector<float>& weights,
                      float xi, float theta_l = 0.025, float theta_r = 2 * M_PI - 0.025);

    std::vector<uint32_t> strategy3_1DoF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                                         const std::vector<float>& weights, const Eigen::Vector3f& axis);
    std::vector<uint32_t> strategy1_3DoF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                                         const std::vector<float>& weights, const Eigen::Vector3f& axis);

    // Init best lower bound by translation only
    float initBestLB(const Eigen::Matrix2Xf& vecs, const std::vector<float>& weights, float xi);

    // Filter inliers after stage 1, refine rc, project to 2D plane
    std::vector<uint32_t> filterInliersProj(const Eigen::Matrix3Xf& src, const Eigen::Matrix3Xf& tgt,
                                            const std::vector<float>& weights, Eigen::Matrix2Xf& midpts,
                                            Eigen::Matrix2Xf& vecs_proj, std::vector<float>& weights_filtered,
                                            Eigen::Vector3f& rc, float xi);
    std::vector<uint32_t> filterInliersStage2(const Eigen::Matrix2Xf& midpts, const Eigen::Matrix2Xf& vecs,
                                              const std::vector<float>& weights, float xi) const;
    std::vector<uint32_t> filterInliersRot(const Eigen::Matrix2Xf& midpts, const Eigen::Matrix2Xf& vecs,
                                           const std::vector<float>& weights, float xi) const;
    std::vector<uint32_t> filterInliersTrans(const Eigen::Matrix2Xf& vecs, const std::vector<float>& weights,
                                             float xi) const;

  private:
    std::array<Eigen::Vector4f, 12> stage1_best;
    float stage2_best, m_rho;
    int m_topk;
    bool m_trans_only, m_rot_near_z;
};

} // namespace gmor
