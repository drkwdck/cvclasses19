/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <random>
#include <ctime>

namespace cvlib
{
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::norm_random(int s, int len_desc)
{
    int max_size = s / 2;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, max_size);
    int x1, y1, x2, y2;

    for (int i = 0; i < len_desc; i++)
    {
        x1 = (int)distribution(generator) % (max_size + 1);
        y1 = (int)distribution(generator) % (max_size + 1);
        x2 = (int)distribution(generator) % (max_size + 1);
        y2 = (int)distribution(generator) % (max_size + 1);
        _points_dict.push_back(cv::Point(x1,y1));
        _points_dict.push_back(cv::Point(x2, y2));
    }

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

    for (auto i = radius; i < image.rows() - radius; i++)
    {
        for(auto j = radius; j < image.cols() - radius; j++)
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

void corner_detector_fast::compute(cv::InputArray img, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat image;
    img.getMat().copyTo(image);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0, 0);
    //Бинарный дескриптор BRIEF
    const int s = 25;
    const int desc_length = 16;

    if (_points_dict.empty())
    {
        norm_random(s, desc_length * 16);
    }

    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_16U);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);
    int half_s = s / 2 + 1;
    cv::copyMakeBorder(image, image, half_s, half_s, half_s, half_s, cv::BORDER_REPLICATE);
    uint16_t* ptr = reinterpret_cast<uint16_t*>(desc_mat.ptr());

    for (auto key_point : keypoints)
    {
        key_point.pt.x += half_s;
        key_point.pt.y += half_s;
        int indx = 0;

        for (int i = 0; i < desc_length; i++)
        {
            uint16_t descrpt = 0;
            
            for (int j = 0; j < 2 * 8; j++)
            {
                uint8_t pix1 = image.at<uint8_t>(key_point.pt + _points_dict[indx]);
                uint8_t pix2 = image.at<uint8_t>(key_point.pt + _points_dict[indx + 1]);
                int bit = (pix1 < pix2);
                descrpt |= bit << (15 - j);
                indx += 2;
            }

            *ptr = descrpt;
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
}
} // namespace cvlib
