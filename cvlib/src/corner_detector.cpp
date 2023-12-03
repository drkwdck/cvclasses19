/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <vector>

#include <ctime>

namespace cvlib
{
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    std::vector<std::pair<int,int>> v = {std::pair<int,int>(0,3),
                                         std::pair<int,int>(1,3),
                                         std::pair<int,int>(2,2),
                                         std::pair<int,int>(3,1),
                                         std::pair<int,int>(3,0),
                                         std::pair<int,int>(3,-1),
                                         std::pair<int,int>(2,-2),
                                         std::pair<int,int>(1,-3),
                                         std::pair<int,int>(-3,0),
                                         std::pair<int,int>(-1,-3),
                                         std::pair<int,int>(-2,-2),
                                         std::pair<int,int>(-3,-1),
                                         std::pair<int,int>(-3,0),
                                         std::pair<int,int>(-3,1),
                                         std::pair<int,int>(-2,2),
                                         std::pair<int,int>(-1,3)};
    keypoints.clear();
    double threshold = 40;
    int N = 12;

    for (auto i = 3; i < image.rows() - 3; i++)
    {
        for(auto j = 3; j < image.cols() - 3; j++)
        {
            int active_pos_count = 0;
            int active_neg_count = 0;

            for (auto idx  : {0, 4, 8, 12})
            {
                auto p = image.getMat().at<unsigned char>(i, j);
                auto p1 = image.getMat().at<unsigned char>(i + v[idx].first,
                                                           j + v[idx].second);
                active_pos_count += (p1 > p + threshold);
                active_neg_count += (p1 < p - threshold);
            }

            if (active_pos_count >= 3 || active_neg_count >= 3)
            {
                active_pos_count = 0;
                active_neg_count = 0;
                bool is_prev_pos = false;

                for (auto d : v)
                {
                    auto p = image.getMat().at<unsigned char>(i, j);
                    auto p1 = image.getMat().at<unsigned char>(i + d.first, j + d.second);
                    
                    if (p1 > p + threshold)
                    {
                        if (!is_prev_pos && active_pos_count > 0)
                        {
                            active_pos_count = 0;
                        }

                        ++active_pos_count;
                        is_prev_pos = true;
                    }


                    if (p1 < p - threshold)
                    {
                        if (is_prev_pos && active_neg_count > 0)
                        {
                            active_neg_count = 0;
                        }

                        ++active_neg_count;
                        is_prev_pos = false;
                    }
                }
                if(active_neg_count >= N || active_pos_count >= N)
                {
                    auto point = cv::KeyPoint(cv::Point2f(j, i), 100);
                    keypoints.push_back(point);
                }
            }
        }
    }
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    std::srand(unsigned(std::time(0)));
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
}
} // namespace cvlib
