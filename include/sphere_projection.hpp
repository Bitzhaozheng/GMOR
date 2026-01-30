/**
 * @file sphere_proj.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Discretization of sphere projection
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include <Eigen/Core>

#include <cmath>

namespace gmor {

template <typename Scalar> class SphereProjBase {
  public:
    SphereProjBase() = default;
    virtual ~SphereProjBase() = default;

    /**
     * @brief Initialize the region as uniformly as possible on the hemisphere.
     *
     * @param k Index of the region, must be less than 3 * 4^{depth}
     * @param depth Depth of quartered regions, and the number of regions is 3 * 4^{depth}
     * @return Eigen::Matrix<Scalar, 2, 2> First column is bottomleft corner, second is topright.
     */
    virtual Eigen::Matrix<Scalar, 2, 2> initRegion(int k, int depth) = 0;

    virtual Eigen::Vector<Scalar, 2> proj(const Eigen::Vector<Scalar, 3>& point) = 0;
    virtual Eigen::Vector<Scalar, 3> projInv(const Eigen::Vector<Scalar, 2>& point) = 0;
    virtual Eigen::Vector<Scalar, 3> projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) = 0;
    virtual Eigen::Vector<Scalar, 3> projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) = 0;
};

/**
 * @brief Stereographic projection.
 *
 * @tparam Scalar
 */
template <typename Scalar> class SphereProjStereoGraphic : public SphereProjBase<Scalar> {
  public:
    SphereProjStereoGraphic() = default;
    ~SphereProjStereoGraphic() override = default;

    Eigen::Matrix<Scalar, 2, 2> initRegion(int k, int depth) override;

    Eigen::Vector<Scalar, 2> proj(const Eigen::Vector<Scalar, 3>& point) override;
    Eigen::Vector<Scalar, 3> projInv(const Eigen::Vector<Scalar, 2>& point) override;
    Eigen::Vector<Scalar, 3> projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) override;
    Eigen::Vector<Scalar, 3> projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) override;
};

/**
 * @brief Miller projection.
 *
 * @tparam Scalar
 */
template <typename Scalar> class SphereProjMiller : public SphereProjBase<Scalar> {
  public:
    SphereProjMiller() = default;
    ~SphereProjMiller() override = default;

    Eigen::Matrix<Scalar, 2, 2> initRegion(int k, int depth) override;

    Eigen::Vector<Scalar, 2> proj(const Eigen::Vector<Scalar, 3>& point) override;
    Eigen::Vector<Scalar, 3> projInv(const Eigen::Vector<Scalar, 2>& point) override;
    Eigen::Vector<Scalar, 3> projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) override;
    Eigen::Vector<Scalar, 3> projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) override;
};

/**
 * @brief Original spherical projection.
 *
 * @tparam Scalar
 */
template <typename Scalar> class SphereProjOri : public SphereProjBase<Scalar> {
  public:
    SphereProjOri() = default;
    ~SphereProjOri() override = default;

    Eigen::Matrix<Scalar, 2, 2> initRegion(int k, int depth) override;

    Eigen::Vector<Scalar, 2> proj(const Eigen::Vector<Scalar, 3>& point) override;
    Eigen::Vector<Scalar, 3> projInv(const Eigen::Vector<Scalar, 2>& point) override;
    Eigen::Vector<Scalar, 3> projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) override;
    Eigen::Vector<Scalar, 3> projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) override;
};

/**
 * @brief Lambert Cylindrical Equal-Area (LCEA) projection.
 *
 * @tparam Scalar
 */
template <typename Scalar> class SphereProjLCEA : public SphereProjBase<Scalar> {
  public:
    SphereProjLCEA() = default;
    ~SphereProjLCEA() override = default;

    Eigen::Matrix<Scalar, 2, 2> initRegion(int k, int depth) override;

    Eigen::Vector<Scalar, 2> proj(const Eigen::Vector<Scalar, 3>& point) override;
    Eigen::Vector<Scalar, 3> projInv(const Eigen::Vector<Scalar, 2>& point) override;
    Eigen::Vector<Scalar, 3> projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) override;
    Eigen::Vector<Scalar, 3> projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) override;
};

/**
 * @brief Lambert Azimuthal Equal-Area (LAEA) projection.
 *
 * @tparam Scalar
 */
template <typename Scalar> class SphereProjLAEA : public SphereProjBase<Scalar> {
  public:
    SphereProjLAEA() = default;
    ~SphereProjLAEA() override = default;

    Eigen::Matrix<Scalar, 2, 2> initRegion(int k, int depth) override;

    Eigen::Vector<Scalar, 2> proj(const Eigen::Vector<Scalar, 3>& point) override;
    Eigen::Vector<Scalar, 3> projInv(const Eigen::Vector<Scalar, 2>& point) override;
    Eigen::Vector<Scalar, 3> projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) override;
    Eigen::Vector<Scalar, 3> projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) override;
};

template <typename Scalar> class SphereProjCube : public SphereProjBase<Scalar> {
  public:
    /**
     * @brief Proj type using cube mapping.
     *
     * eCubeX: (1, Y, Z)
     * eCubeY: (X, 1, Z)
     * eCubeZ: (X, Y, 1)
     */
    enum CubeType { eCubeX = 0, eCubeY = 1, eCubeZ = 2 };

    SphereProjCube() : m_cube_type(eCubeZ) {}
    explicit SphereProjCube(unsigned int cube_type) : m_cube_type(cube_type % 3) {}
    ~SphereProjCube() override = default;

    Eigen::Matrix<Scalar, 2, 2> initRegion(int k, int depth) override;

    Eigen::Vector<Scalar, 2> proj(const Eigen::Vector<Scalar, 3>& point) override;
    Eigen::Vector<Scalar, 3> projInv(const Eigen::Vector<Scalar, 2>& point) override;
    Eigen::Vector<Scalar, 3> projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) override;
    Eigen::Vector<Scalar, 3> projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) override;

  private:
    unsigned char m_cube_type;
};

/*****************Implementation******************/

/*****************SphereProjStereoGraphic******************/
template <typename Scalar> Eigen::Matrix<Scalar, 2, 2> SphereProjStereoGraphic<Scalar>::initRegion(int k, int depth) {
    if (depth < 1)
        depth = 1;
    int num_regions_dim = std::pow(2, depth);
    int num_regions_face = num_regions_dim * num_regions_dim;
    int num_regions = 3 * num_regions_face;
    if (k < 0)
        k = 0;
    if (k >= num_regions)
        k = num_regions - 1;

    int region_id = k % num_regions_face;
    int face_id = k / num_regions_face;
    int x_id = region_id % num_regions_dim;
    int y_id = region_id / num_regions_dim;
    Scalar stride_x = 2.0 * M_PI / 3.0 / num_regions_dim;
    Scalar stride_y = 2.0 / num_regions_dim;

    Eigen::Matrix<Scalar, 2, 2> region;
    region(0, 0) = (-1.0 + face_id * 2 / 3.0) * M_PI + x_id * stride_x;
    region(0, 1) = (-1.0 + face_id * 2 / 3.0) * M_PI + (x_id + 1) * stride_x;
    region(1, 0) = y_id * stride_y;
    region(1, 1) = (y_id + 1) * stride_y;
    return region;
}

template <typename Scalar>
Eigen::Vector<Scalar, 2> SphereProjStereoGraphic<Scalar>::proj(const Eigen::Vector<Scalar, 3>& point) {
    Scalar lon = std::atan2(point(1), point(0));
    Scalar lat_inv = std::acos(point(2) / point.norm());
    Scalar y = 2 * std::tan(0.5 * lat_inv);
    Eigen::Vector<Scalar, 2> xy(lon, y);
    return xy;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjStereoGraphic<Scalar>::projInv(const Eigen::Vector<Scalar, 2>& point) {
    Scalar lat_inv = 2 * std::atan(point(1) / 2); // lat in [0, pi/2]
    Scalar x_rot = std::cos(point(0)) * std::sin(lat_inv);
    Scalar y_rot = std::sin(point(0)) * std::sin(lat_inv);
    Scalar z_rot = std::cos(lat_inv);
    Eigen::Vector<Scalar, 3> position(x_rot, y_rot, z_rot);
    return position;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3>
SphereProjStereoGraphic<Scalar>::projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) {
    Eigen::Vector<Scalar, 2> c = (region.col(0) + region.col(1)) * 0.5;
    return projInv(c);
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjStereoGraphic<Scalar>::projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) {
    return projInv(region.col(0));
}
/*****************SphereProjStereoGraphic end******************/

/*****************SphereProjMiller******************/
template <typename Scalar> Eigen::Matrix<Scalar, 2, 2> SphereProjMiller<Scalar>::initRegion(int k, int depth) {
    if (depth < 1)
        depth = 1;
    int num_regions_dim = std::pow(2, depth);
    int num_regions_face = num_regions_dim * num_regions_dim;
    int num_regions = 3 * num_regions_face;
    if (k < 0)
        k = 0;
    if (k >= num_regions)
        k = num_regions - 1;

    // Hemishphere is divided into 3 faces along phi-coordinate
    int region_id = k % num_regions_face;
    int face_id = k / num_regions_face;
    int x_id = region_id % num_regions_dim;
    int y_id = region_id / num_regions_dim;
    Scalar stride_x = 2.0 * M_PI / 3.0 / num_regions_dim;
    Scalar stride_y = 1.25 * std::log(std::tan(9.0 * M_PI / 20.0)) / num_regions_dim;

    Eigen::Matrix<Scalar, 2, 2> region;
    region(0, 0) = (-1.0 + face_id * 2 / 3.0) * M_PI + x_id * stride_x;
    region(0, 1) = (-1.0 + face_id * 2 / 3.0) * M_PI + (x_id + 1) * stride_x;
    region(1, 0) = y_id * stride_y;
    region(1, 1) = (y_id + 1) * stride_y;
    return region;
}

template <typename Scalar>
Eigen::Vector<Scalar, 2> SphereProjMiller<Scalar>::proj(const Eigen::Vector<Scalar, 3>& point) {
    Scalar lon = std::atan2(point(1), point(0));
    Scalar lat = std::asin(point(2) / point.norm());
    // Eq. (11-2) in the manual
    Scalar y = 1.25 * std::log(std::tan(0.25 * M_PI + 0.4 * lat));
    Eigen::Vector<Scalar, 2> xy(lon, y);
    return xy;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjMiller<Scalar>::projInv(const Eigen::Vector<Scalar, 2>& point) {
    // Eq. (11-6) in the manual
    Scalar lat = std::atan(std::exp(0.8 * point(1))) * 2.5 - 0.625 * M_PI; // lat in [0, pi/2]
    Scalar x_rot = std::cos(point(0)) * std::cos(lat);
    Scalar y_rot = std::sin(point(0)) * std::cos(lat);
    Scalar z_rot = std::sin(lat);
    Eigen::Vector<Scalar, 3> position(x_rot, y_rot, z_rot);
    return position;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjMiller<Scalar>::projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) {
    Eigen::Vector<Scalar, 2> phiz_c = (region.col(0) + region.col(1)) * 0.5;
    return projInv(phiz_c);
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjMiller<Scalar>::projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) {
    // The farthest point is at the minimum z-coordinate according to TRDE
    return projInv(region.col(0));
}
/*****************SphereProjMiller end******************/

/*****************SphereProjOri******************/
template <typename Scalar> Eigen::Matrix<Scalar, 2, 2> SphereProjOri<Scalar>::initRegion(int k, int depth) {
    if (depth < 1)
        depth = 1;
    int num_regions_dim = std::pow(2, depth);
    int num_regions_face = num_regions_dim * num_regions_dim;
    int num_regions = 3 * num_regions_face;
    if (k < 0)
        k = 0;
    if (k >= num_regions)
        k = num_regions - 1;

    // Hemishphere is divided into 3 faces along phi-coordinate
    int region_id = k % num_regions_face;
    int face_id = k / num_regions_face;
    int phi_id = region_id % num_regions_dim;
    int theta_id = region_id / num_regions_dim;
    Scalar stride_phi = 2.0 * M_PI / 3.0 / num_regions_dim;
    Scalar stride_theta = M_PI_2 / num_regions_dim;

    Eigen::Matrix<Scalar, 2, 2> region;
    region(0, 0) = (-1.0 + face_id * 2 / 3.0) * M_PI + phi_id * stride_phi;
    region(0, 1) = (-1.0 + face_id * 2 / 3.0) * M_PI + (phi_id + 1) * stride_phi;
    region(1, 0) = theta_id * stride_theta;
    region(1, 1) = (theta_id + 1) * stride_theta;
    return region;
}

template <typename Scalar> Eigen::Vector<Scalar, 2> SphereProjOri<Scalar>::proj(const Eigen::Vector<Scalar, 3>& point) {
    Scalar phi = std::atan2(point(1), point(0));
    Scalar theta = std::asin(std::abs(point(2)) / point.norm());
    Eigen::Vector<Scalar, 2> phitheta(phi, theta);
    return phitheta;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjOri<Scalar>::projInv(const Eigen::Vector<Scalar, 2>& point) {
    Eigen::Vector<Scalar, 3> position(std::cos(point(0)) * std::cos(point(1)), std::sin(point(0)) * std::cos(point(1)),
                                      std::sin(point(1)));
    return position;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjOri<Scalar>::projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) {
    Eigen::Vector<Scalar, 2> phitheta_c = (region.col(0) + region.col(1)) * 0.5;
    return projInv(phitheta_c);
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjOri<Scalar>::projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) {
    return projInv(region.col(1));
}
/*****************SphereProjOri end******************/

/*****************SphereProjLCEA******************/
template <typename Scalar> Eigen::Matrix<Scalar, 2, 2> SphereProjLCEA<Scalar>::initRegion(int k, int depth) {
    if (depth < 1)
        depth = 1;
    int num_regions_dim = std::pow(2, depth);
    int num_regions_face = num_regions_dim * num_regions_dim;
    int num_regions = 3 * num_regions_face;
    if (k < 0)
        k = 0;
    if (k >= num_regions)
        k = num_regions - 1;

    // Hemishphere is divided into 3 faces along phi-coordinate
    int region_id = k % num_regions_face;
    int face_id = k / num_regions_face;
    int phi_id = region_id % num_regions_dim;
    int z_id = region_id / num_regions_dim;
    Scalar stride_phi = 2.0 * M_PI / 3.0 / num_regions_dim;
    Scalar stride_z = 1.0 / num_regions_dim;

    Eigen::Matrix<Scalar, 2, 2> region;
    region(0, 0) = (-1.0 + face_id * 2 / 3.0) * M_PI + phi_id * stride_phi;
    region(0, 1) = (-1.0 + face_id * 2 / 3.0) * M_PI + (phi_id + 1) * stride_phi;
    region(1, 0) = z_id * stride_z;
    region(1, 1) = (z_id + 1) * stride_z;
    return region;
}

template <typename Scalar>
Eigen::Vector<Scalar, 2> SphereProjLCEA<Scalar>::proj(const Eigen::Vector<Scalar, 3>& point) {
    Scalar phi = std::atan2(point(1), point(0));
    Scalar z = point(2);
    Eigen::Vector<Scalar, 2> phiz(phi, z);
    return phiz;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjLCEA<Scalar>::projInv(const Eigen::Vector<Scalar, 2>& point) {
    Scalar r = std::sqrt(1.0 - point(1) * point(1));
    Scalar x_rot = r * std::cos(point(0));
    Scalar y_rot = r * std::sin(point(0));
    Eigen::Vector<Scalar, 3> position(x_rot, y_rot, point(1));
    return position;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjLCEA<Scalar>::projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) {
    Eigen::Vector<Scalar, 2> phiz_c = (region.col(0) + region.col(1)) * 0.5;
    return projInv(phiz_c);
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjLCEA<Scalar>::projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) {
    // The farthest point is at the maximum z-coordinate
    return projInv(region.col(1));
}
/*****************SphereProjLCEA end******************/

/*****************SphereProjLAEA******************/
template <typename Scalar> Eigen::Matrix<Scalar, 2, 2> SphereProjLAEA<Scalar>::initRegion(int k, int depth) {
    if (depth < 1)
        depth = 1;
    int num_regions_dim = std::pow(2, depth);
    int num_regions_face = num_regions_dim * num_regions_dim;
    int num_regions = 3 * num_regions_face;
    if (k < 0)
        k = 0;
    if (k >= num_regions)
        k = num_regions - 1;

    // Hemishphere is divided into 3 faces along phi-coordinate
    int region_id = k % num_regions_face;
    int face_id = k / num_regions_face;
    int phi_id = region_id % num_regions_dim;
    int z_id = region_id / num_regions_dim;
    Scalar stride_phi = 2.0 * M_PI / 3.0 / num_regions_dim;
    Scalar stride_rho = std::sqrt(2.0) / num_regions_dim;

    Eigen::Matrix<Scalar, 2, 2> region;
    region(0, 0) = (-1.0 + face_id * 2 / 3.0) * M_PI + phi_id * stride_phi;
    region(0, 1) = (-1.0 + face_id * 2 / 3.0) * M_PI + (phi_id + 1) * stride_phi;
    region(1, 0) = z_id * stride_rho;
    region(1, 1) = (z_id + 1) * stride_rho;
    return region;
}

template <typename Scalar>
Eigen::Vector<Scalar, 2> SphereProjLAEA<Scalar>::proj(const Eigen::Vector<Scalar, 3>& point) {
    // Eq. (24-7) in the manual
    Scalar phi = std::atan2(point(1), point(0));
    Scalar lat = std::asin(point(2) / point.norm());
    Scalar rho = 2 * std::sin(M_PI_4 - lat * 0.5);
    Eigen::Vector<Scalar, 2> phirho(phi, rho);
    return phirho;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjLAEA<Scalar>::projInv(const Eigen::Vector<Scalar, 2>& point) {
    Scalar lat_inv = 2 * std::asin(0.5 * point(1)); // pi / 2 - lat
    Scalar x_rot = std::sin(lat_inv) * std::cos(point(0));
    Scalar y_rot = std::sin(lat_inv) * std::sin(point(0));
    Eigen::Vector<Scalar, 3> position(x_rot, y_rot, std::cos(lat_inv));
    return position;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjLAEA<Scalar>::projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) {
    Eigen::Vector<Scalar, 2> c = (region.col(0) + region.col(1)) * 0.5;
    return projInv(c);
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjLAEA<Scalar>::projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) {
    return projInv(region.col(1));
}
/*****************SphereProjLAEA end******************/

/*****************SphereProjCube******************/
template <typename Scalar> Eigen::Matrix<Scalar, 2, 2> SphereProjCube<Scalar>::initRegion(int k, int depth) {
    if (depth < 1)
        depth = 1;
    int num_regions_dim = std::pow(2, depth);
    int num_regions_face = num_regions_dim * num_regions_dim;
    int num_regions = 3 * num_regions_face;
    if (k < 0)
        k = 0;
    if (k >= num_regions)
        k = num_regions - 1;

    m_cube_type = k / num_regions_face;
    int region_id = k % num_regions_face;
    int x_id = region_id % num_regions_dim;
    int y_id = region_id / num_regions_dim;
    Scalar stride = 2.0 / num_regions_dim;

    Eigen::Matrix<Scalar, 2, 2> region;
    region(0, 0) = -1.0 + x_id * stride;
    region(0, 1) = -1.0 + (x_id + 1) * stride;
    region(1, 0) = -1.0 + y_id * stride;
    region(1, 1) = -1.0 + (y_id + 1) * stride;
    return region;
}

template <typename Scalar>
Eigen::Vector<Scalar, 2> SphereProjCube<Scalar>::proj(const Eigen::Vector<Scalar, 3>& point) {
    Eigen::Vector<Scalar, 2> xy;
    xy(0) = point((m_cube_type + 2) % 3) / point(m_cube_type % 3);
    xy(1) = point((m_cube_type + 1) % 3) / point(m_cube_type % 3);
    return xy;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjCube<Scalar>::projInv(const Eigen::Vector<Scalar, 2>& point) {
    Eigen::Vector<Scalar, 3> position;
    position((m_cube_type + 2) % 3) = point(0);
    position((m_cube_type + 1) % 3) = point(1);
    position(m_cube_type % 3) = 1.0;

    position.normalize();
    return position;
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjCube<Scalar>::projInvRegionCenter(const Eigen::Matrix<Scalar, 2, 2>& region) {
    Eigen::Vector<Scalar, 2> xy_c = (region.col(0) + region.col(1)) * 0.5;
    return projInv(xy_c);
}

template <typename Scalar>
Eigen::Vector<Scalar, 3> SphereProjCube<Scalar>::projInvRegionMax(const Eigen::Matrix<Scalar, 2, 2>& region) {
    Eigen::Vector<Scalar, 2> xy_max;
    xy_max(0) = std::fabs(region(0, 0)) < std::fabs(region(0, 1)) ? region(0, 0) : region(0, 1);
    xy_max(1) = std::fabs(region(1, 0)) < std::fabs(region(1, 1)) ? region(1, 0) : region(1, 1);
    return projInv(xy_max);
}
/*****************SphereProjCube end******************/

} // namespace gmor
