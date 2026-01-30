/**
 * @file TRDE.h
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Unofficial implementation of TR-DE "Deterministic Point Cloud Registration via Novel Transformation
 * Decomposition" (Wen Chen et al., CVPR 2022)
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include "registration_base.hpp"

#include <cmath>

namespace trde {

class TRDESolver : public gmor::RegistrationBnBBase<float> {
  public:
    TRDESolver();
    ~TRDESolver() override;

    Eigen::Matrix4f solve(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target,
                          const std::vector<std::tuple<int, int, float>>& correspondences) override;

    using Ptr = std::shared_ptr<TRDESolver>;
    using Ptr_u = std::unique_ptr<TRDESolver>;

  protected:
    // Stage I: Searching for (2+1) DOF in Section 4.2
    // Discretization of the hemisphere based on Miller's method in Section 4.1
    // ry_r is the upper bound of ry in Table 1 (supplementary material), d is the extreme case of translation
    void stage1_2and1DOF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target, float rx_l = -M_PI,
                         float rx_r = M_PI, float ry_l = 0.0, float ry_r = 1.25 * log(tan(9.0 * M_PI / 20.0)),
                         float d_l = -2.0, float d_r = 2.0);
    // Stage II: Searching for Remaining (1+2) DOF in Section 4.3
    void stage2_1and2DOF(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target, float phi_l = -M_PI,
                         float phi_r = M_PI, float h_l = 0.0, float h_r = 2.0);

    void innerBnB(const Eigen::Matrix3Xf& source, const Eigen::Matrix3Xf& target, const Eigen::Vector<float, 3>& rc,
                  const float& dc, const float* phi_lu, const float* h_lu, float* theta_lu);

    // Filter the current inliers between stage 1 and 2
    void filterInliersStage1(Eigen::Matrix3Xf& source, Eigen::Matrix3Xf& target);

  private:
    // Search results
    float stage1_best[3], stage2_best[3];
    size_t m_numInliers; // Number of inliers after filtering
};

} // namespace trde
