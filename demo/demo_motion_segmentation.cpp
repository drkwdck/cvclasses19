/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

cvlib::motion_segmentation* mseg = nullptr;

void handleThresholdChanged(int val, void*)
{
    mseg->setVarThreshold(val);
}

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    mseg = new cvlib::motion_segmentation();
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";
    const auto min_wnd = "min";
    const auto max_wnd = "max";
    const auto diff_wnd = "diff";
    const auto bg_wnd = "bg";

    int threshold = 50;
    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    cv::namedWindow(max_wnd);
    cv::namedWindow(min_wnd);
    cv::namedWindow(diff_wnd);
    cv::namedWindow(bg_wnd);
    cv::createTrackbar("th", demo_wnd, &threshold, 255, handleThresholdChanged);

    cv::Mat frame;
    cv::Mat frame_mseg;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        mseg->apply(frame, frame_mseg);
        if (!frame_mseg.empty())
        {
            cv::Mat bg;
            mseg->getBackgroundImage(bg);
            cv::imshow(demo_wnd, frame_mseg);
            cv::imshow(max_wnd, mseg->getMax());
            cv::imshow(diff_wnd, mseg->getDiff());
            cv::imshow(min_wnd, mseg->getMin());
            cv::imshow(bg_wnd, bg);
        }
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);
    cv::destroyWindow(max_wnd);
    cv::destroyWindow(min_wnd);
    cv::destroyWindow(bg_wnd);
    cv::destroyWindow(diff_wnd);

    return 0;
}
