#include "demo_pipeline.h"

#include "exp/exp_io.hpp"
#include "exp/exp_utils.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <glog/logging.h>

#include <chrono>

namespace demo {

int demo_simulation(const SimDataConfig& config) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZ>);
    if (demo::readPointCloud<pcl::PointXYZ>(config.src_file_name, src) < 0)
        return -1;

    LOG(INFO) << "Source cloud size: " << src->size();

    // Randomly downsample the source cloud
    pcl::RandomSample<pcl::PointXYZ> random_filter;
    random_filter.setInputCloud(src);
    random_filter.setSample(config.num_samples);
    random_filter.filter(*src);

    float scale = demo::normalizePointCloud(src);

    LOG(INFO) << "Downsampled source cloud size: " << src->size();
    LOG(INFO) << "Scale to " << scale;

    // Random transformation
    Eigen::Matrix3f rot_mat;
    Eigen::Vector3f translation_mat;
    demo::genRandRotMat(rot_mat);
    demo::genRandTranslation(translation_mat, eSPHERE, 0.0, 5.0);
    Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
    transform.linear() = rot_mat;
    transform.translation() = translation_mat;
    Eigen::AngleAxisf rot_gt(rot_mat);

    LOG(INFO) << "Initial transformation: \n" << transform.matrix();
    pcl::transformPointCloud(*src, *tgt, transform.matrix());
    LOG(INFO) << "Initial rotation axis: \n(" << rot_gt.axis().transpose() << "): " << rot_gt.angle() * 180.0 / M_PI
              << " (deg)";

    // Add noise and outliers after normalization
    constexpr float chi_3_95 = 2.796;
    demo::addNoiseandOutliers(tgt, config.noise_bound / chi_3_95, config.outlier_ratio, 0.0, 10.0);

    // Visualize the initial clouds
    pcl::visualization::PCLVisualizer viewer("Init");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_color_handler(src, 255, 170, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_color_handler(tgt, 0, 0, 255);
    viewer.setBackgroundColor(255, 255, 255);
    viewer.addPointCloud(src, src_color_handler, "src");
    viewer.addPointCloud(tgt, tgt_color_handler, "tgt");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "src");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "tgt");
    viewer.spinOnce(100);

    // Transform to Eigen::Matrix3Xf
    std::vector<std::tuple<int, int, float>> corrs;
    corrs.reserve(src->size());
    for (int i = 0; i < (int)src->size(); ++i) {
        corrs.emplace_back(i, i, 1.0f);
    }
    Eigen::Matrix3Xf src_points = src->getMatrixXfMap().topRows<3>();
    Eigen::Matrix3Xf tgt_points = tgt->getMatrixXfMap().topRows<3>();

    Eigen::Matrix4f transform_result;
    double elapsed_ms = globalRegistration(src_points, tgt_points, corrs, config, transform_result);

    double rotation_error = getRotDist(transform_result.topLeftCorner<3, 3>(), transform.linear(), true);
    double translation_error = (transform_result.topRightCorner<3, 1>() - transform.translation()).norm();

    LOG(INFO) << "Transformation:\n" << transform_result;
    LOG(INFO) << "Elapsed time: " << elapsed_ms << " ms";
    LOG(INFO) << "Rotation error: " << rotation_error << " (deg)";
    LOG(INFO) << "Translation error: " << translation_error;

    // Visualization of result
    pcl::transformPointCloud(*src, *src, transform_result);
    pcl::visualization::PCLVisualizer viewer_result("Result");
    viewer_result.setBackgroundColor(1.0, 1.0, 1.0);
    viewer_result.addPointCloud(src, src_color_handler, "src");
    viewer_result.addPointCloud(tgt, tgt_color_handler, "tgt");
    viewer_result.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "src");
    viewer_result.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "tgt");
    viewer_result.spin();

    return 0;
}

int demo_realdata(const RealDataConfig& config, bool icp_refine) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_origin(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_origin(new pcl::PointCloud<pcl::PointXYZ>);
    if (demo::readPointCloud<pcl::PointXYZ>(config.src_file_name, src_origin) < 0)
        return -1;
    if (demo::readPointCloud<pcl::PointXYZ>(config.tgt_file_name, tgt_origin) < 0)
        return -1;

    // Visualization of the original clouds
    pcl::visualization::PCLVisualizer viewer("Original");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_o_color_handler(src_origin, 255, 170, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_o_color_handler(tgt_origin, 0, 0, 255);
    viewer.setBackgroundColor(255, 255, 255);
    viewer.addCoordinateSystem(10.0 * config.voxel_size);
    viewer.addPointCloud(src_origin, src_o_color_handler, "src");
    viewer.addPointCloud(tgt_origin, tgt_o_color_handler, "tgt");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "src");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "tgt");
    viewer.spinOnce(100);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    // Voxel grid downsample
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_o(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_o(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(config.voxel_size, config.voxel_size, config.voxel_size);
    voxel_filter.setInputCloud(src_origin);
    voxel_filter.filter(*src_o);
    voxel_filter.setInputCloud(tgt_origin);
    voxel_filter.filter(*tgt_o);

    // Generate FPFH correspondences
    pcl::Correspondences corrs;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features_o(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tgt_features_o(new pcl::PointCloud<pcl::FPFHSignature33>);
    demo::computeFPFH(src_o, src_features_o, config.radius_normal, config.radius_feature);
    demo::computeFPFH(tgt_o, tgt_features_o, config.radius_normal, config.radius_feature);

    // Random sample
    pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tgt_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    if (config.random_sample) {
        std::vector<int> src_indices(src_features_o->size());
        std::iota(src_indices.begin(), src_indices.end(), 0);
        std::vector<int> tgt_indices(tgt_features_o->size());
        std::iota(tgt_indices.begin(), tgt_indices.end(), 0);
        // Set random seed to 0 for reproductive results (negative means using std::random_device)
        std::vector<int> src_sample_indices = sampleN(src_indices, config.num_samples, 0);
        std::vector<int> tgt_sample_indices = sampleN(tgt_indices, config.num_samples, 0);
        pcl::copyPointCloud(*src_o, src_sample_indices, *src);
        pcl::copyPointCloud(*src_features_o, src_sample_indices, *src_features);
        pcl::copyPointCloud(*tgt_o, tgt_sample_indices, *tgt);
        pcl::copyPointCloud(*tgt_features_o, tgt_sample_indices, *tgt_features);
    } else {
        src = src_o;
        src_features = src_features_o;
        tgt = tgt_o;
        tgt_features = tgt_features_o;
    }

    // Test selfmade descriptors
    matchFPFH(src_features, tgt_features, corrs, config, true);

    LOG(INFO) << "Source: " << src->size() << ", Target: " << tgt->size();
    LOG(INFO) << "Number of FPFH correspondences: " << corrs.size();

    // Start registration
    Eigen::Matrix4f transform_result;
    double elapsed_ms = demo::globalRegistrationPCL(src, tgt, corrs, config, transform_result);

    // An entire coarse-to-fine pipeline must contain an ICP-based refinement
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    if (icp_refine) {
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(src);
        icp.setInputTarget(tgt_o);
        icp.setTransformationEpsilon(1e-6);
        icp.setMaxCorrespondenceDistance(config.noise_bound);
        icp.setMaximumIterations(50);
        icp.align(*src_trans, transform_result);
        transform_result = icp.getFinalTransformation();
    } else {
        pcl::transformPointCloud(*src, *src_trans, transform_result);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    LOG(INFO) << "Transformation:\n" << transform_result;
    LOG(INFO) << "Registration time: " << elapsed_ms << " ms";
    elapsed_ms = std::chrono::duration_cast<std::chrono::duration<int, std::micro>>(end - start).count() * 1e-3;
    LOG(INFO) << "Total Downsampling-FPFH-Matching-Registration time: " << elapsed_ms << " ms";

    // Save transformation matrix
    std::ofstream ofs("Rt.txt");
    ofs << transform_result.matrix() << std::endl;
    ofs.close();

    // Visualize the downsampled clouds
    pcl::visualization::PCLVisualizer viewer_downsampled("Downsampled");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_color_handler(src, 255, 170, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_color_handler(tgt, 0, 0, 255);
    viewer_downsampled.setBackgroundColor(255, 255, 255);
    viewer_downsampled.addPointCloud(src, src_color_handler, "src");
    viewer_downsampled.addPointCloud(tgt, tgt_color_handler, "tgt");
    viewer_downsampled.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "src");
    viewer_downsampled.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "tgt");
    viewer_downsampled.spinOnce(100);

    // Visualize result
    pcl::visualization::PCLVisualizer viewer_result("Result");
    viewer_result.setBackgroundColor(255, 255, 255);
    viewer_result.addPointCloud(src_trans, src_color_handler, "src");
    viewer_result.addPointCloud(tgt, tgt_color_handler, "tgt");
    viewer_result.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "src");
    viewer_result.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "tgt");
    viewer_result.spin();

    return 0;
}

int run_exp(const std::string& data_path, const RealDataConfig& config) {
    switch (config.data_set) {
    case e3DMatch:
    case e3DLoMatch:
        return exp_3DMatch(data_path, config);
    case eKITTI:
        return exp_KITTI(data_path, config);
    default:
        return -1;
    }
}

} // namespace demo
