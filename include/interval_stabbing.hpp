/**
 * @file interval_stabbing.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Interval stabbing algorithm with gradually convergence
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace gmor {

// first: border, second: weight
template <typename BorderT, typename WeightT> using Interval = std::pair<BorderT, WeightT>;
template <typename BorderT, typename WeightT> using Intervals = std::vector<Interval<BorderT, WeightT>>;
// 0: border, 1: weight, 2: index
template <typename BorderT, typename WeightT> using CenterIndice = std::tuple<BorderT, WeightT, uint32_t>;
template <typename BorderT, typename WeightT> using CenterIndices = std::vector<CenterIndice<BorderT, WeightT>>;

/**
 * @brief Interval stabbing for correspondences with constant length
 *
 * @tparam BorderT float or double
 * @tparam WeightT float, double, integer
 * @param centers Interval centers (center, weight)
 * @param len Interval length
 * @return WeightT Maximum sum of weights
 */
template <typename BorderT, typename WeightT>
WeightT intervalStabbingConstLen(Intervals<BorderT, WeightT>& centers, BorderT len);

/**
 * @brief Interval stabbing for correspondences
 *
 * @tparam BorderT float or double
 * @tparam WeightT float, double, integer
 * @param intervals Intervals (border, weight)
 * @return WeightT Maximum sum of weights
 */
template <typename BorderT, typename WeightT> WeightT intervalStabbing(Intervals<BorderT, WeightT>& intervals);

/**
 * @brief Interval stabbing and find the band of peak
 *
 * @tparam BorderT float or double
 * @tparam WeightT float, double, integer
 * @param intervals Intervals (border, weight)
 * @param bound_max Left and right borders of peak
 * @param best_lb Threshold of lower bound
 * @param rho Convergence ratio
 * @return WeightT Maximum sum of weights
 */
template <typename BorderT, typename WeightT>
WeightT intervalStabbing(Intervals<BorderT, WeightT>& intervals, BorderT* bound_max, const WeightT& best_lb,
                         BorderT rho = 0.25);

/**
 * @brief Filter the indices of correspondences with maximum sum of weights
 *
 * @tparam BorderT float or double
 * @tparam WeightT float, double, integer
 * @param center_indices Interval centers (border, weight, indices)
 * @return std::vector<uint32_t> Filtered indices
 */
template <typename BorderT, typename WeightT>
std::vector<uint32_t> intervalStabbingFilterIndices(CenterIndices<BorderT, WeightT>& center_indices, BorderT len);

/*****************Implementation******************/

template <typename BorderT, typename WeightT>
WeightT intervalStabbingConstLen(Intervals<BorderT, WeightT>& centers, BorderT len) {
    if (centers.empty())
        return 0;
    WeightT max_weight = 0, sum_weight = 0;
    std::sort(centers.begin(), centers.end());
    size_t j = 0;
    for (const auto& center : centers) {
        while (j < centers.size() && centers[j].first <= center.first + len)
            sum_weight += centers[j++].second;
        max_weight = std::max(max_weight, sum_weight);
        sum_weight -= center.second;
    }
    return max_weight;
}

template <typename BorderT, typename WeightT> WeightT intervalStabbing(Intervals<BorderT, WeightT>& intervals) {
    if (intervals.empty())
        return 0;
    WeightT max_weight = 0, sum_weight = 0;

    // Sort the vector by ascending coordinates from less to greater.
    std::sort(intervals.begin(), intervals.end());

    for (const auto& interval : intervals) {
        // Accumulated weight of intervals
        sum_weight += interval.second;
        // Peak of accumulated weight
        max_weight = std::max(max_weight, sum_weight);
    }
    return max_weight;
}

template <typename BorderT, typename WeightT>
WeightT intervalStabbing(Intervals<BorderT, WeightT>& intervals, BorderT* bound_max, const WeightT& best_lb,
                         BorderT rho) {
    if (intervals.empty())
        return 0;
    WeightT max_weight = 0, sum_weight = 0;

    // Sort the vector by ascending coordinates from less to greater.
    std::sort(intervals.begin(), intervals.end());

    size_t index_max = 0;
    for (size_t i = 0; i < intervals.size(); ++i) {
        // Accumulated weight of intervals
        sum_weight += intervals[i].second;
        // Peak of accumulated weight
        if (sum_weight > max_weight) {
            max_weight = sum_weight;
            index_max = i;
        }
    }

    // Upper bound is less than current best lower bound, it should be pruned
    if (max_weight <= best_lb) {
        return max_weight;
    }

    // Find the left bound of peak
    sum_weight = max_weight;
    for (int i = index_max; i >= 0; --i) {
        sum_weight -= intervals[i].second;
        if (sum_weight - best_lb < (max_weight - best_lb) * rho) {
            bound_max[0] = intervals[i].first;
            break;
        }
    }

    // Right bound
    sum_weight = max_weight;
    for (size_t i = index_max + 1; i < intervals.size(); ++i) {
        sum_weight += intervals[i].second;
        if (sum_weight - best_lb < (max_weight - best_lb) * rho) {
            bound_max[1] = intervals[i].first;
            break;
        }
    }
    return max_weight;
}

template <typename BorderT, typename WeightT>
std::vector<uint32_t> intervalStabbingFilterIndices(CenterIndices<BorderT, WeightT>& center_indices, BorderT len) {
    std::vector<uint32_t> indices;
    if (center_indices.empty()) {
        return indices;
    }
    WeightT max_weight = 0, sum_weight = 0;

    std::sort(center_indices.begin(), center_indices.end());
    size_t j = 0, index_max = 0;

    for (size_t i = 0; i < center_indices.size(); ++i) {
        // Accumulated weight of intervals
        BorderT right_bd = std::get<0>(center_indices[i]) + len;
        while (j < center_indices.size() && std::get<0>(center_indices[j]) <= right_bd)
            sum_weight += std::get<1>(center_indices[j++]);
        // Peak of accumulated weight
        if (sum_weight > max_weight) {
            index_max = i;
            max_weight = sum_weight;
        }
        sum_weight -= std::get<1>(center_indices[i]);
    }

    // Filter the indices of peak weight
    BorderT right_bd = std::get<0>(center_indices[index_max]) + len;
    for (size_t i = index_max; i < center_indices.size() && std::get<0>(center_indices[i]) <= right_bd; ++i) {
        indices.push_back(std::get<2>(center_indices[i]));
    }
    return indices;
}

} // namespace gmor
