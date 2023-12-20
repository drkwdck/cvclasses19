/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors,
                                      std::vector<std::vector<cv::DMatch>>& matches,
                                      int k,
                                      cv::InputArrayOfArrays masks,
                                      bool compactResult)
{

    if (trainDescCollection.empty())
        return;

    cv::Mat q_desc;
    queryDescriptors.getMat().convertTo(q_desc, CV_32F);
    cv::Mat t_desc;
    trainDescCollection[0].convertTo(t_desc, CV_32F);
    matches.resize(q_desc.rows);

    for (int i = 0; i < q_desc.rows; ++i)
    {
        cv::Mat t = t_desc - cv::repeat(q_desc.row(i), t_desc.rows, 1);
        cv::multiply(t, t, t);
        cv::Mat res;
        cv::reduce(t, res, 1, CV_REDUCE_SUM);
        double first_min_val;
        double secon_min_val;
        cv::Point first_loc_min_point;
        cv::Point second_loc_min_point;
        minMaxLoc(res, &first_min_val, nullptr, &first_loc_min_point, nullptr);
        res.at<float>(first_loc_min_point) = FLT_MAX;
        minMaxLoc(res, &secon_min_val, nullptr, &second_loc_min_point, nullptr);
        double ssd_ratio = first_min_val / secon_min_val;

        if(ssd_ratio <= ratio_)
        {
            matches[i].emplace_back(i, first_loc_min_point.y, first_min_val);
        }
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors,
                                         std::vector<std::vector<cv::DMatch>>& matches,
                                         float maxDistance,
                                         cv::InputArrayOfArrays masks,
                                         bool compactResult)
{
    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);

    for (auto& match : matches)
    {
        if (!match.empty())
        {
            if (match[0].distance > maxDistance)
            {
                match.clear();
            }
        }
    }
}
} // namespace cvlib
