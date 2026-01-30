
#include "demo_pipeline.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <omp.h>

DEFINE_string(path, "../Dataset/3DMatch", "Dataset path");
DEFINE_string(src, "../data/cloud_bin_0.ply", "Source point cloud path");
DEFINE_string(tgt, "../data/cloud_bin_7.ply", "Target point cloud path");

// Simulation or real data
DEFINE_bool(sim, false, "Data type: true for simulation; or false for real data");
DEFINE_bool(exp, false, "Data type: true for 3DMatch or KITTI datasets; or false for demo data");
// Simulation data parameters, reused in KITTI dataset evaluation
DEFINE_int32(num_samples, 8000, "Number of samples");
DEFINE_double(outlier_ratio, 0.95, "Outliers ratio");
// Real data parameters, default for 3DMatch dataset
DEFINE_string(datatype, "3DMatch", "Dataset type: 3DMatch/3DLoMatch/KITTI");
DEFINE_string(feature, "FPFH", "Feature descriptor: FPFH/FCGF/Predator");
DEFINE_double(voxel_size, 0.05, "Downsampling voxel size");
DEFINE_double(radius_normal, 0.10, "FPFH normal radius");
DEFINE_double(radius_feature, 0.30, "FPFH radius");
DEFINE_bool(random_sample, false, "Randomly sample points");
DEFINE_bool(icp_refine, false, "ICP refinement after global registration, only for demo");
// Feature matching parameters
DEFINE_int32(knn, 40, "Number of nearest neighbors in feature space for weights");
DEFINE_double(df, 0.01, "Distance factor of softmax matching in feature space");

// Registration method
DEFINE_string(reg, "GMOR", "Registration method: TRDE, GMOR");
// Common parameters in BnB registration
DEFINE_int32(num_threads, 12, "Number of OpenMP threads");
DEFINE_double(noise_bound, 0.10, "Noise bound threshold");
DEFINE_double(branch_eps, 5e-2, "Minimum branch resolution (rad) in BnB");
DEFINE_double(bound_eps, 1e-3, "Upper lower bound threshold in BnB");
// Hyperparameters of GMOR
DEFINE_int32(topk, 12, "Top-k rotation axes");
DEFINE_double(rho, 0.25, "Convergence ratio");
DEFINE_bool(rot_near_z, false, "Whether rotation axis is near z-axis, used for KITTI dataset");

int main(int argc, char* argv[]) {
    std::string usage = "Usage: RegistrationFactory [options]";
    google::SetUsageMessage(usage);
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    // Global setting of OpenMP, not recommended to use in large projects
    omp_set_num_threads(FLAGS_num_threads);

    // Enable colored log
    FLAGS_colorlogtostderr = true;

    demo::RegistrationMethod method;
    if (FLAGS_reg == "TRDE") {
        method = demo::RegistrationMethod::eTRDE;
    } else if (FLAGS_reg == "GMOR") {
        method = demo::RegistrationMethod::eGMOR;
    } else {
        LOG(ERROR) << "Invalid registration method: " << FLAGS_reg;
        return -1;
    }

    demo::RegConfig config{FLAGS_src,
                           method,
                           (float)FLAGS_noise_bound,
                           FLAGS_num_threads,
                           FLAGS_num_samples,
                           (float)FLAGS_branch_eps,
                           (float)FLAGS_bound_eps,
                           FLAGS_topk,
                           (float)FLAGS_rho,
                           FLAGS_rot_near_z};

    if (FLAGS_sim) {
        // Simulation
        demo::SimDataConfig config_sim(config);
        config_sim.outlier_ratio = FLAGS_outlier_ratio;
        return demo::demo_simulation(config_sim);
    } else {
        demo::RealDataConfig config_real(config);
        // Feature matching parameters
        config_real.knn = FLAGS_knn;
        config_real.df = FLAGS_df;
        // Preprocess parameters for raw point cloud (Downsampling, normal estimation, and FPFH estimation)
        config_real.voxel_size = FLAGS_voxel_size;
        config_real.radius_normal = FLAGS_radius_normal;
        config_real.radius_feature = FLAGS_radius_feature;

        if (FLAGS_exp) {
            // Select feature
            if (FLAGS_feature == "FPFH") {
                config_real.feature = demo::FeatureType::eFPFH;
            } else if (FLAGS_feature == "FCGF") {
                config_real.feature = demo::FeatureType::eFCGF;
            } else if (FLAGS_feature == "Predator") {
                config_real.feature = demo::FeatureType::ePredator;
            } else {
                LOG(ERROR) << "Invalid feature type: " << FLAGS_feature;
                return -1;
            }

            // Dataset
            if (FLAGS_datatype == "KITTI") {
                config_real.data_set = demo::DataSet::eKITTI;
            } else if (FLAGS_datatype == "3DMatch") {
                config_real.data_set = demo::DataSet::e3DMatch;
            } else if (FLAGS_datatype == "3DLoMatch") {
                config_real.data_set = demo::DataSet::e3DLoMatch;
            } else {
                LOG(ERROR) << "Invalid dataset type: " << FLAGS_datatype;
                return -1;
            }
            return demo::run_exp(FLAGS_path, config_real);
        } else {
            config_real.tgt_file_name = FLAGS_tgt;
            // Whether to randomly sample num_samples points in demo
            config_real.random_sample = FLAGS_random_sample;
            return demo::demo_realdata(config_real, FLAGS_icp_refine);
        }
    }

    return 0;
}