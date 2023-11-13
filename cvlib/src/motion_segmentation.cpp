/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <iostream>

namespace cvlib
{
motion_segmentation::motion_segmentation() : _mu(0), _counter(0)
{
}
void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double)
{
    if (_counter > 100)
        _counter = 0;

    if (_image.empty())
        return;
    cv::Mat image = _image.getMat().clone();
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    if (!_counter)
    {
        image.copyTo(_max);
        image.copyTo(_min);
        _diff = cv::Mat::zeros(image.size(), CV_8UC1);
        _prevImage = cv::Mat::zeros(image.size(), CV_8UC1);
    }
    
    _mu = 0;
    cv::Mat fgmask = cv::Mat::zeros(image.size(), CV_8UC1);

    for (int i = 0; i < image.size().width; i++)
    {
        for (int j = 0; j < image.size().height; j++)
        {
            uchar pix = image.at<uchar>(j, i);
            _max.at<uchar>(j, i) = std::max(_max.at<uchar>(j, i), pix);
            _min.at<uchar>(j, i) = std::min(_min.at<uchar>(j, i), pix);

            if (_counter > 0)
            {
                uchar diff = (uchar)std::abs(double(pix) - double(_prevImage.at<uchar>(j, i)));
                _diff.at<uchar>(j, i) = std::max(_diff.at<uchar>(j, i), diff);
            }

            _prevImage.at<uchar>(j, i) = pix;
        }
    }

    cv::Mat tmp;
    cv::sort(_diff, tmp, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
    _mu = tmp.at<uchar>(tmp.size().height / 2, tmp.size().width / 2);

    for (int i = 0; i < image.size().width; i++)
    {
        for (int j = 0; j < image.size().height; j++)
        {
            double pix = (double)image.at<uchar>(j, i);

            if (std::abs((double)_max.at<uchar>(j, i) - pix) > (double)_threshold / 100 * _mu)
            {
                fgmask.at<uchar>(j, i) = 255;
            } else if (std::abs((double)_min.at<uchar>(j, i) - pix) > (double)_threshold / 100 * _mu) 
            {
                fgmask.at<uchar>(j, i) = 255;
            }
            else
            {
                fgmask.at<uchar>(j, i) = 0;
            }
        }
    }

    cv::Mat mask;
    cv::bitwise_not(fgmask, mask);
    cv::bitwise_or(_image, _image, bg_model_, mask);

    fgmask.copyTo(_fgmask);
    _counter++;
}
} // namespace cvlib
