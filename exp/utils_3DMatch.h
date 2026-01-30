// This file is modified from http://redwood-data.org/indoor/fileformat.html

#pragma once

#include <Eigen/Core>

#include <map>
#include <vector>

struct FramedTransformation {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id2_;
    int frame_;
    Eigen::Matrix4d transformation_;
    FramedTransformation(int id2, int f, Eigen::Matrix4d& t) : id2_(id2), frame_(f), transformation_(t) {}
};

struct RGBDTrajectory {
    std::map<int, std::vector<FramedTransformation, Eigen::aligned_allocator<FramedTransformation>>> data_;

    void LoadFromFile(std::string& filename) {
        data_.clear();
        int id1, id2, frame;
        Eigen::Matrix4d trans;
        FILE* f = fopen(filename.c_str(), "r");
        if (f != nullptr) {
            char buffer[1024];
            while (fgets(buffer, 1024, f) != nullptr) {
                if (strlen(buffer) > 0 && buffer[0] != '#') {
                    sscanf(buffer, "%d %d %d", &id1, &id2, &frame);
                    if (fgets(buffer, 1024, f) != nullptr)
                        sscanf(buffer, "%lf %lf %lf %lf", &trans(0, 0), &trans(0, 1), &trans(0, 2), &trans(0, 3));
                    if (fgets(buffer, 1024, f) != nullptr)
                        sscanf(buffer, "%lf %lf %lf %lf", &trans(1, 0), &trans(1, 1), &trans(1, 2), &trans(1, 3));
                    if (fgets(buffer, 1024, f) != nullptr)
                        sscanf(buffer, "%lf %lf %lf %lf", &trans(2, 0), &trans(2, 1), &trans(2, 2), &trans(2, 3));
                    if (fgets(buffer, 1024, f) != nullptr)
                        sscanf(buffer, "%lf %lf %lf %lf", &trans(3, 0), &trans(3, 1), &trans(3, 2), &trans(3, 3));
                    if (data_.find(id1) == data_.end())
                        data_.emplace(
                            id1, std::vector<FramedTransformation, Eigen::aligned_allocator<FramedTransformation>>());
                    data_[id1].emplace_back(id2, frame, trans);
                }
            }
            fclose(f);
        } else {
            printf("Error: cannot open file %s\n", filename.c_str());
        }
    }
    void SaveToFile(std::string& filename) {
        FILE* f = fopen(filename.c_str(), "w");
        for (auto& [id1, data_vec] : data_) {
            for (auto& data : data_vec) {
                Eigen::Matrix4d& trans = data.transformation_;
                fprintf(f, "%d\t%d\t%d\n", id1, data.id2_, data.frame_);
                fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(0, 0), trans(0, 1), trans(0, 2), trans(0, 3));
                fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(1, 0), trans(1, 1), trans(1, 2), trans(1, 3));
                fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(2, 0), trans(2, 1), trans(2, 2), trans(2, 3));
                fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(3, 0), trans(3, 1), trans(3, 2), trans(3, 3));
            }
        }
        fclose(f);
    }
};
