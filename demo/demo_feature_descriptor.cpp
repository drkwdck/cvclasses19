/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

void hist(cv::Mat& input, cv::Mat& output)
{
    int hist_size = 256;
    float range[] = {0, 256};
    const float *hist_range = {range};
    cv::Mat hist;
    calcHist(&input, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, true, false);
    normalize(hist, hist, 0, input.rows, cv::NORM_MINMAX, -1, cv::Mat());
    int padding = 10;
    int total_padding = padding * (256 - 1);
    int max_width = (800 - total_padding) / 256;
    cv::Mat haming_hist(500, 800, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < 256; i++)
    {
        rectangle(haming_hist,  cv::Point((max_width + padding) * i, 500),
                  cv::Point((max_width + padding) * i + max_width, 500 - cvRound(hist.at<float>(i * (hist_size / 256)))),
                  cv::Scalar(255, 0, 0), cv::FILLED);
    }

    haming_hist.copyTo(output);
}

int demo_feature_descriptor(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";
    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    cv::Mat frame;
    auto detector_a = cv::ORB::create();
    auto detector_b = cvlib::corner_detector_fast::create();
    std::vector<cv::KeyPoint> corners;
    cv::Mat descriptors;
    cv::Mat descriptors_orb;

    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector_b->detect(frame, corners); // \todo use your detector (detector_b)
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));
        utils::put_fps_text(frame, fps);
        const auto fontScale = 0.5;
        const auto thickness = 1;
        static const cv::Point text_org_point = {frame.rows / 8, frame.cols / 8};
        std::stringstream ss;
        ss << "detected: " << std::fixed << corners.size();

        cv::putText(frame, ss.str(), text_org_point, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 0), thickness, 8, false);
        cv::imshow(demo_wnd, frame);

        pressed_key = cv::waitKey(30);
        // \todo draw histogram of SSD distribution for all descriptors instead of dumping into the file
        if (pressed_key == ' ') // space
        {
            detector_a->compute(frame, corners, descriptors_orb);
            detector_b->compute(frame, corners, descriptors);

            cv::Mat d_hamming = cv::Mat(descriptors.rows, descriptors.cols, CV_16U);
            cv::Mat haming_hist;
            cv::Mat descriptors_orb16bit = cv::Mat(descriptors.rows, descriptors.cols, CV_16U, cv::Scalar(0,0,0));
            cv::FileStorage file("descriptor.json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

            for (int i = 0; i < descriptors_orb.rows; i++)
            {
                for (int j = 0; j < descriptors_orb.cols; j+=2)
                {
                    descriptors_orb16bit.at<uint16_t>(i, j / 2) = (descriptors_orb.at<uint8_t>(i, j) << 8) | descriptors_orb.at<uint8_t>(i, j + 1);
                }
            }

            cv::bitwise_xor(descriptors, descriptors_orb16bit, d_hamming);
            hist(d_hamming, haming_hist);
            namedWindow("hist", cv::WINDOW_AUTOSIZE);
            imshow("Histogram", haming_hist);
            file << "orb" << descriptors_orb;
            file << "detector_b" << descriptors;

            std::cout << "Dump descriptors complete! \n";
        }

        std::cout << "Feature points: " << corners.size() << "\r";
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}