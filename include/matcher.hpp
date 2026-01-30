/**
 * @file matcher.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Match features and find weighted correspondences compatible with PCL.
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>

#include <nanoflann.hpp>

#include <omp.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace gmor {

// For comparison of inlier counting based algorithms
// #define EN_COUNTING

enum MatchType {
    eSourcetoTarget = 0x00, // Source to target matching
    eCross = 0x01,          // Cross matching
    eRejectTopk = 0x02,     // Reject correspondences that are not in top-k nearest lists
    eUniformWeight = 0x04,  // Uniform weight (1.0) or non-uniform
    eSoftmax = 0x08         // If non-uniform weight, set to softmax weight or squared dist weight
};

// FeatureDescriptor and KdTreeFLANN are used for classes that are not instantiated in pcl::KdTreeFLANN<>.
// Feature descriptor copied from pcl::FPFHSignature33
template <int N> struct FeatureDescriptor {
    float histogram[N] = {0.0f};

    static constexpr int descriptorSize() { return N; }

    inline constexpr FeatureDescriptor() = default;
};

// Modified nanoflann implementation based on pcl::KdTreeFLANN
template <typename FeatureSignature, typename Dist = nanoflann::metric_L2> class KdTreeFLANN {
  public:
    using FeatureCloudPtr = typename pcl::PointCloud<FeatureSignature>::Ptr;
    using kdTree =
        nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf, FeatureSignature::descriptorSize(), Dist, false>;

    KdTreeFLANN() = default;
    ~KdTreeFLANN() = default;

    void setInputCloud(const FeatureCloudPtr& cloud);

    int nearestKSearch(const FeatureSignature& point, int k, std::vector<int>& k_indices,
                       std::vector<float>& k_sqr_dists) const;

    using Ptr = std::shared_ptr<KdTreeFLANN>;

  private:
    std::shared_ptr<kdTree> flann_index_;
    Eigen::MatrixXf data_;
};

template <typename FeatureSignature, class SearchMethod = KdTreeFLANN<FeatureSignature>> class FeatureMatcher {
  public:
    using FeatureCloudPtr = typename pcl::PointCloud<FeatureSignature>::Ptr;
    using KdTreePtr = typename SearchMethod::Ptr;

    FeatureMatcher();
    FeatureMatcher(int k, float df);
    ~FeatureMatcher() = default;

    void setNeighbors(int k);

    void setdf(float df);

    /**
     * @brief Match features and find correspondences.
     *
     * @param src_features Input source features
     * @param tgt_features Input target features
     * @param correspondences Output correspondences
     * @param type Matching type. Default is cross-matching and non-uniform mean square weight.
     */
    void match(const FeatureCloudPtr& src_features, const FeatureCloudPtr& tgt_features,
               pcl::Correspondences& correspondences, int type = eCross);
    // For Predator
    void matchwithSaliency(const FeatureCloudPtr& src_features, const FeatureCloudPtr& tgt_features,
                           pcl::Correspondences& correspondences, std::vector<float>& src_saliency_scores,
                           std::vector<float>& tgt_saliency_scores, int type = eCross);

    /**
     * @brief Softmax weighting function
     *
     * @param dists The sorted squared distances estimated by FLANN
     * @param temperature_2 2 * df^2, where df is the descriptor distance factor
     * @param alpha Scale factor, generally set to squared norm of the feature descriptor
     * @param beta Bias
     * @return float Softmax weight
     */
    static float softmaxWeight(const std::vector<float>& dists, float temperature_2, float alpha = 1.0f,
                               float beta = 0.0f);
    static float squaredDistWeight(const std::vector<float>& dists, float alpha = 1.0f, float beta = 0.0f);

  private:
    int m_knn;
    float m_df;
    FeatureCloudPtr m_src_features;
    FeatureCloudPtr m_tgt_features;
    KdTreePtr m_src_kdtree, m_tgt_kdtree;
};

/*****************Implementation******************/

/*****************kdTreeFLANN******************/
template <typename FeatureSignature, typename Dist>
void KdTreeFLANN<FeatureSignature, Dist>::setInputCloud(const FeatureCloudPtr& cloud) {
    data_.resize(FeatureSignature::descriptorSize(), cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        data_.col(i) =
            Eigen::Map<const Eigen::Vector<float, FeatureSignature::descriptorSize()>>(cloud->at(i).histogram);
    }
    // flann_index_->index->buildIndex() is called in constructor
    flann_index_ = std::make_shared<kdTree>(FeatureSignature::descriptorSize(), std::cref(data_), 15);
}

template <typename FeatureSignature, typename Dist>
int KdTreeFLANN<FeatureSignature, Dist>::nearestKSearch(const FeatureSignature& point, int k,
                                                        std::vector<int>& k_indices,
                                                        std::vector<float>& k_sqr_dists) const {
    if (k <= 0)
        k = 1;
    k_indices.resize(k);
    k_sqr_dists.resize(k);

    // For compatibility with int indices in PCL
    nanoflann::KNNResultSet<float, int, int> resultSet(k);
    resultSet.init(&k_indices[0], &k_sqr_dists[0]);
    // index has been renamed to index_ since version 1.5.0
#if NANOFLANN_VERSION >= 0x150
    flann_index_->index_->findNeighbors(resultSet, &point.histogram[0], nanoflann::SearchParameters());
#else
    flann_index_->index->findNeighbors(resultSet, &point.histogram[0], nanoflann::SearchParams());
#endif

    return k;
}

/*****************kdTreeFLANN end******************/

/*****************FeatureMatcher******************/
template <typename FeatureSignature, class SearchMethod>
FeatureMatcher<FeatureSignature, SearchMethod>::FeatureMatcher() : FeatureMatcher(40, 0.01) {}

template <typename FeatureSignature, class SearchMethod>
FeatureMatcher<FeatureSignature, SearchMethod>::FeatureMatcher(int k, float df) : m_knn(k), m_df(df) {
    m_src_features = nullptr;
    m_tgt_features = nullptr;
    m_src_kdtree.reset(new SearchMethod);
    m_tgt_kdtree.reset(new SearchMethod);
}

template <typename FeatureSignature, class SearchMethod>
void FeatureMatcher<FeatureSignature, SearchMethod>::setNeighbors(int k) {
    m_knn = k;
}

template <typename FeatureSignature, class SearchMethod>
void FeatureMatcher<FeatureSignature, SearchMethod>::setdf(float df) {
    m_df = df;
}

template <typename FeatureSignature, class SearchMethod>
void FeatureMatcher<FeatureSignature, SearchMethod>::match(const FeatureCloudPtr& src_features,
                                                           const FeatureCloudPtr& tgt_features,
                                                           pcl::Correspondences& correspondences, int type) {

    // FIXME: The kdtree will be rebuilt only if pointer changes
    if (m_src_features == nullptr || src_features != m_src_features) {
        m_src_kdtree->setInputCloud(src_features);
        m_src_features = src_features;
    }
    if (m_tgt_features == nullptr || tgt_features != m_tgt_features) {
        m_tgt_kdtree->setInputCloud(tgt_features);
        m_tgt_features = tgt_features;
    }

    // 2 * temperature because $d^2 \sim 2(1-\cos(\theta))$ between normalized vectors relavant to cosine similarity
    float temperature_2 = 2 * m_df * m_df;

    // Find correspondences using kdtree-based nearest neighbor search
    int k = type & eUniformWeight ? 1 : this->m_knn;
    if (type & eCross)
        correspondences.resize(src_features->size() + tgt_features->size(), pcl::Correspondence(0, 0, 0));
    else
        correspondences.resize(src_features->size(), pcl::Correspondence(0, 0, 0));

#pragma omp parallel for default(none) firstprivate(k, type, temperature_2) shared(correspondences) schedule(dynamic)
    for (int i = 0; i < (int)m_src_features->size(); ++i) {
        // source to target
        // Only nearestKSearch can be called in multi-threading
        std::vector<int> nn_i(k);
        std::vector<float> nn_dists2(k);
        m_tgt_kdtree->nearestKSearch(m_src_features->at(i), k, nn_i, nn_dists2);
        if (type & eRejectTopk) {
            std::vector<int> nn_j(k);
            std::vector<float> nn_j_dists2(k);
            m_src_kdtree->nearestKSearch(m_tgt_features->at(nn_i[0]), k, nn_j, nn_j_dists2);
            if (nn_j_dists2[k - 1] < nn_dists2[0])
                continue;
        }
        Eigen::Map<const Eigen::Vector<float, FeatureSignature::descriptorSize()>> src_hist(
            m_src_features->at(i).histogram);
        float weight;
        if (type & eUniformWeight) {
            weight = 1.0f;
        } else if (type & eSoftmax) {
            weight = softmaxWeight(nn_dists2, temperature_2, src_hist.squaredNorm() + 1e-6);
        } else { // squared dist
            weight = squaredDistWeight(nn_dists2, src_hist.squaredNorm() + 1e-6);
        }
        // Make sure that correspondence[i].index_query = i
        if (!std::isnan(weight) && weight > 1e-6) {
            correspondences[i] = pcl::Correspondence(i, nn_i[0], weight);
        } else {
            correspondences[i] = pcl::Correspondence(i, nn_i[0], 1e-6); // Positive weight
        }
    }

    // Cross matching
    if (type & eCross) {
#pragma omp parallel for default(none) firstprivate(k, type, temperature_2) shared(correspondences) schedule(dynamic)
        for (int i = 0; i < (int)m_tgt_features->size(); ++i) {
            std::vector<int> nn_i(k);
            std::vector<float> nn_dists2(k);
            m_src_kdtree->nearestKSearch(m_tgt_features->at(i), k, nn_i, nn_dists2);
            if (type & eRejectTopk) {
                std::vector<int> nn_j(k);
                std::vector<float> nn_j_dists2(k);
                m_tgt_kdtree->nearestKSearch(m_src_features->at(nn_i[0]), k, nn_j, nn_j_dists2);
                if (nn_j_dists2[k - 1] < nn_dists2[0])
                    continue;
            }
            Eigen::Map<const Eigen::Vector<float, FeatureSignature::descriptorSize()>> tgt_hist(
                m_tgt_features->at(i).histogram);
            float weight;
            if (type & eUniformWeight) {
                weight = 1.0f;
            } else if (type & eSoftmax) {
                weight = softmaxWeight(nn_dists2, temperature_2, tgt_hist.squaredNorm() + 1e-6);
            } else { // squared dist
                weight = squaredDistWeight(nn_dists2, tgt_hist.squaredNorm() + 1e-6);
            }
            if (!std::isnan(weight) && weight > 1e-6) {
#ifndef EN_COUNTING
                if (i == correspondences[nn_i[0]].index_match) {
                    correspondences[nn_i[0]].weight += weight;
                    correspondences[i + m_src_features->size()] = pcl::Correspondence(nn_i[0], i, 0);
                } else
#endif
                {
                    correspondences[i + m_src_features->size()] = pcl::Correspondence(nn_i[0], i, weight);
                }
            } else {
                correspondences[i + m_src_features->size()] = pcl::Correspondence(nn_i[0], i, 1e-6); // Positive weight
            }
        }
    }
    // Remove invalid corrs
    correspondences.erase(
        std::remove_if(correspondences.begin(), correspondences.end(),
                       [](const pcl::Correspondence& c) { return c.weight < 2.01e-6 || std::isnan(c.weight); }),
        correspondences.end());
    correspondences.shrink_to_fit();
}

template <typename FeatureSignature, class SearchMethod>
void FeatureMatcher<FeatureSignature, SearchMethod>::matchwithSaliency(
    const FeatureCloudPtr& src_features, const FeatureCloudPtr& tgt_features, pcl::Correspondences& correspondences,
    std::vector<float>& src_saliency_scores, std::vector<float>& tgt_saliency_scores, int type) {

    float temperature_2 = 2 * m_df * m_df;

    constexpr int dim_with_saliency = FeatureSignature::descriptorSize() + 1;
    typename pcl::PointCloud<FeatureDescriptor<dim_with_saliency>>::Ptr src_features_with_saliency(
        new pcl::PointCloud<FeatureDescriptor<dim_with_saliency>>);
    typename pcl::PointCloud<FeatureDescriptor<dim_with_saliency>>::Ptr tgt_features_with_saliency(
        new pcl::PointCloud<FeatureDescriptor<dim_with_saliency>>);
    for (int i = 0; i < (int)src_features->size(); ++i) {
        FeatureDescriptor<dim_with_saliency> feature_with_saliency;
        std::copy(src_features->at(i).histogram, src_features->at(i).histogram + FeatureSignature::descriptorSize(),
                  feature_with_saliency.histogram);
        feature_with_saliency.histogram[FeatureSignature::descriptorSize()] =
            std::sqrt(-temperature_2 * std::log(src_saliency_scores[i]));
        src_features_with_saliency->push_back(feature_with_saliency);
    }
    for (int i = 0; i < (int)tgt_features->size(); ++i) {
        FeatureDescriptor<dim_with_saliency> feature_with_saliency;
        std::copy(tgt_features->at(i).histogram, tgt_features->at(i).histogram + FeatureSignature::descriptorSize(),
                  feature_with_saliency.histogram);
        feature_with_saliency.histogram[FeatureSignature::descriptorSize()] =
            std::sqrt(-temperature_2 * std::log(tgt_saliency_scores[i]));
        tgt_features_with_saliency->push_back(feature_with_saliency);
    }
    KdTreeFLANN<FeatureDescriptor<dim_with_saliency>> src_kdtree_flann, tgt_kdtree_flann;
    src_kdtree_flann.setInputCloud(src_features_with_saliency);
    tgt_kdtree_flann.setInputCloud(tgt_features_with_saliency);

    // Find correspondences using kdtree-based nearest neighbor search
    int k = type & eUniformWeight ? 1 : this->m_knn;
    if (type & eCross)
        correspondences.resize(src_features->size() + tgt_features->size(), pcl::Correspondence(0, 0, 0));
    else
        correspondences.resize(src_features->size(), pcl::Correspondence(0, 0, 0));

#pragma omp parallel for default(none) firstprivate(k, type, temperature_2)                                            \
    shared(src_features, tgt_features, src_kdtree_flann, tgt_kdtree_flann, src_saliency_scores, tgt_saliency_scores,   \
               correspondences) schedule(dynamic)
    for (int i = 0; i < (int)src_features->size(); ++i) {
        // source to target
        FeatureDescriptor<dim_with_saliency> feature_with_saliency;
        std::copy(src_features->at(i).histogram, src_features->at(i).histogram + FeatureSignature::descriptorSize(),
                  feature_with_saliency.histogram);
        feature_with_saliency.histogram[FeatureSignature::descriptorSize()] = 0.0f;
        std::vector<int> nn_i(k);
        std::vector<float> nn_dists2(k);
        tgt_kdtree_flann.nearestKSearch(feature_with_saliency, k, nn_i, nn_dists2);
        if (type & eRejectTopk) {
            FeatureDescriptor<dim_with_saliency> feature_with_saliency_j;
            std::copy(tgt_features->at(nn_i[0]).histogram,
                      tgt_features->at(nn_i[0]).histogram + FeatureSignature::descriptorSize(),
                      feature_with_saliency_j.histogram);
            feature_with_saliency_j.histogram[FeatureSignature::descriptorSize()] = 0.0f;
            std::vector<int> nn_j(k);
            std::vector<float> nn_j_dists2(k);
            src_kdtree_flann.nearestKSearch(feature_with_saliency_j, k, nn_j, nn_j_dists2);
            if (nn_j_dists2[k - 1] < nn_dists2[0])
                continue;
        }

        float weight;
        if (type & eUniformWeight) {
            weight = src_saliency_scores[i] * tgt_saliency_scores[nn_i[0]];
        } else if (type & eSoftmax) {
            weight = src_saliency_scores[i] *
                     softmaxWeight(nn_dists2, temperature_2, 1.0f, std::log(tgt_saliency_scores[nn_i[0]]));
        } else { // squared dist
            weight = src_saliency_scores[i] * tgt_saliency_scores[nn_i[0]] * squaredDistWeight(nn_dists2);
        }
        // Make sure that correspondence[i].index_query = i
        if (!std::isnan(weight) && weight > 1e-6) {
            correspondences[i] = pcl::Correspondence(i, nn_i[0], weight);
        } else {
            correspondences[i] = pcl::Correspondence(i, nn_i[0], 1e-6); // Positive weight
        }
    }

    // Cross matching
    if (type & eCross) {
#pragma omp parallel for default(none) firstprivate(k, type, temperature_2)                                            \
    shared(src_features, tgt_features, src_kdtree_flann, tgt_kdtree_flann, src_saliency_scores, tgt_saliency_scores,   \
               correspondences) schedule(dynamic)
        for (int i = 0; i < (int)tgt_features->size(); ++i) {
            FeatureDescriptor<dim_with_saliency> feature_with_saliency;
            std::copy(tgt_features->at(i).histogram, tgt_features->at(i).histogram + FeatureSignature::descriptorSize(),
                      feature_with_saliency.histogram);
            feature_with_saliency.histogram[FeatureSignature::descriptorSize()] = 0.0f;
            std::vector<int> nn_i(k);
            std::vector<float> nn_dists2(k);
            src_kdtree_flann.nearestKSearch(feature_with_saliency, k, nn_i, nn_dists2);
            if (type & eRejectTopk) {
                FeatureDescriptor<dim_with_saliency> feature_with_saliency_j;
                std::copy(src_features->at(nn_i[0]).histogram,
                          src_features->at(nn_i[0]).histogram + FeatureSignature::descriptorSize(),
                          feature_with_saliency_j.histogram);
                feature_with_saliency_j.histogram[FeatureSignature::descriptorSize()] = 0.0f;
                std::vector<int> nn_j(k);
                std::vector<float> nn_j_dists2(k);
                tgt_kdtree_flann.nearestKSearch(feature_with_saliency_j, k, nn_j, nn_j_dists2);
                if (nn_j_dists2[k - 1] < nn_dists2[0])
                    continue;
            }
            float weight;
            if (type & eUniformWeight) {
                weight = tgt_saliency_scores[i] * src_saliency_scores[nn_i[0]];
            } else if (type & eSoftmax) {
                weight = tgt_saliency_scores[i] *
                         softmaxWeight(nn_dists2, temperature_2, 1.0f, std::log(src_saliency_scores[nn_i[0]]));
            } else { // squared dist
                weight = tgt_saliency_scores[i] * src_saliency_scores[nn_i[0]] * squaredDistWeight(nn_dists2);
            }
            if (!std::isnan(weight) && weight > 1e-6) {
#ifndef EN_COUNTING
                if (i == correspondences[nn_i[0]].index_match) {
                    correspondences[nn_i[0]].weight += weight;
                    correspondences[i + src_features->size()] = pcl::Correspondence(nn_i[0], i, 0);
                } else
#endif
                {
                    correspondences[i + src_features->size()] = pcl::Correspondence(nn_i[0], i, weight);
                }
            } else {
                correspondences[i + src_features->size()] = pcl::Correspondence(nn_i[0], i, 1e-6); // Positive weight
            }
        }
    }
    // Remove invalid corrs
    correspondences.erase(
        std::remove_if(correspondences.begin(), correspondences.end(),
                       [](const pcl::Correspondence& c) { return c.weight < 2.01e-6 || std::isnan(c.weight); }),
        correspondences.end());
    correspondences.shrink_to_fit();
}

template <typename FeatureSignature, class SearchMethod>
float FeatureMatcher<FeatureSignature, SearchMethod>::softmaxWeight(const std::vector<float>& dists,
                                                                    float temperature_2, float alpha, float beta) {
    if (dists.empty())
        return 0.0f;
    if (alpha < 1e-6)
        alpha = 1e-6;
    float temp = 1.0f + std::exp((dists[0] / alpha - 2.0) / temperature_2 + beta);
    for (size_t i = 1; i < dists.size(); i++) {
        float d = std::exp((dists[0] - dists[i]) / (temperature_2 * alpha));
        if (d < 1e-6)
            break;
        temp += d;
    }
    return 1.0f / temp;
}

template <typename FeatureSignature, class SearchMethod>
float FeatureMatcher<FeatureSignature, SearchMethod>::squaredDistWeight(const std::vector<float>& dists, float alpha,
                                                                        float beta) {
    if (dists.empty() || dists[0] >= alpha)
        return 0.0f;
    if (alpha < 1e-6)
        alpha = 1e-6;
    float temp = beta;
    for (size_t i = 0; i < dists.size() && dists[i] < alpha; i++) {
        temp += 1.0f - dists[i] / alpha;
    }
    return (1.0f - dists[0] / alpha) / temp;
}

/*****************FeatureMatcher end******************/

} // namespace gmor
