#include "cvlib.hpp"
#include <opencv2/opencv.hpp>

namespace
{
void split_image(cv::Mat image, double stddev, int& area_num)
{
    cv::Mat mean;
    cv::Mat dev;

    // Если площадь изображения меньше порога, установить весь кластер в одно значение
    if (image.cols * image.rows < 16)
    {
        image.setTo(area_num);
        area_num = area_num + 1;
        return;
    }

    // Определение среднего значения и стандартного отклонения в пределах области
    cv::meanStdDev(image, mean, dev);

    // Если стандартное отклонение меньше порога, установить весь кластер в одно значение
    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(area_num);
        area_num = area_num + 1;
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;

    // Рекурсивное разделение изображения на четыре части
    split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev, area_num);
    split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev, area_num);
    split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev, area_num);
    split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev, area_num);
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    cv::Mat res(image.size(), CV_32SC1);
    image.assignTo(res, CV_32SC1);
    int cluster_num = 0;
    split_image(res, stddev, cluster_num);
    // Инициализация списков соседей и объединенных кластеров
    std::vector<std::list<int>> neighbour_list(cluster_num + 1, std::list<int>());
    std::vector<std::list<int>> concatenated_list(cluster_num + 1, std::list<int>());

    for (size_t i = 0; i < cluster_num; ++i)
    {
        concatenated_list[i].push_back(i);
    }

    // Инициализация вспомогательных переменных для хранения границ кластеров
    std::vector<int> higth_bound(cluster_num + 1, 1e9);
    std::vector<int> botom_bound(cluster_num + 1, 0);
    std::vector<int> left_bound(cluster_num + 1, 1e9);
    std::vector<int> right_bound(cluster_num + 1, 0);

    // Цикл для обработки изображения и определения соседних кластеров
    for (int i = 0; i < res.rows; ++i)
    {
        for (int j = 0; j < res.cols; ++j)
        {
            int neighbour = res.at<int>(i, j);
            higth_bound[neighbour] = std::min(higth_bound[neighbour], i);
            botom_bound[neighbour] = std::max(botom_bound[neighbour], i);
            left_bound[neighbour] = std::min(left_bound[neighbour], j);
            right_bound[neighbour] = std::max(right_bound[neighbour], j);

            if (i + 1 < res.rows)
            {
                int possible_neighbour = res.at<int>(i + 1, j);
                if (neighbour != possible_neighbour &&
                    (std::find(neighbour_list[neighbour].begin(), neighbour_list[neighbour].end(), possible_neighbour) == neighbour_list[neighbour].end()))
                    neighbour_list[neighbour].push_back(possible_neighbour);
            }

            if (j + 1 < res.cols)
            {
                int possible_neighbour = res.at<int>(i, j + 1);
                if (neighbour != possible_neighbour &&
                    (std::find(neighbour_list[neighbour].begin(), neighbour_list[neighbour].end(), possible_neighbour) == neighbour_list[neighbour].end()))
                    neighbour_list[neighbour].push_back(possible_neighbour);
            }
        }
    }

    bool merged = true;
    std::set<int> extended_areas;

    // Цикл для слияния соседних кластеров схожих по статистике
    while (merged)
    {
        merged = false;
        for (size_t i = 0; i < cluster_num; ++i)
        {
            if (extended_areas.find(i) != extended_areas.end())
                continue;

            cv::Mat mean, dev;
            int area_sum = 0;
            int area_count = 0;

            for (auto joined_area : concatenated_list[i])
            {
                area_sum += cv::sum(image(cv::Range(higth_bound[joined_area], botom_bound[joined_area]),
                                          cv::Range(left_bound[joined_area], right_bound[joined_area])))[0];
                area_count += (botom_bound[joined_area] - higth_bound[joined_area]) * (right_bound[joined_area] - left_bound[joined_area]);
            }

            for (auto j : neighbour_list[i])
            {
                if (extended_areas.find(j) != extended_areas.end() || i == j)
                    continue;

                int current_area_sum = 0;
                int current_area_count = 0;

                for (auto current_joined_area : concatenated_list[j])
                {
                    current_area_sum += cv::sum(image(cv::Range(higth_bound[current_joined_area], botom_bound[current_joined_area]),
                                                      cv::Range(left_bound[current_joined_area], right_bound[current_joined_area])))[0];
                    current_area_count += (botom_bound[current_joined_area] - higth_bound[current_joined_area]) *
                                          (right_bound[current_joined_area] - left_bound[current_joined_area]);
                }

                int mean_total = area_sum / area_count;
                int mean_current = current_area_sum / current_area_count;
                int mean_val = (area_sum + current_area_sum) / (area_count + current_area_count);
                const int mean_bound = 40;

                if ((abs(mean_val - mean_total) > mean_bound) || (abs(mean_val - mean_current) > mean_bound))
                    continue;

                double area_std_dev = 0;

                for (auto reg : concatenated_list[i])
                {
                    auto tmp = image(cv::Range(higth_bound[reg], botom_bound[reg]), cv::Range(left_bound[reg], right_bound[reg])) - mean_val;
                    area_std_dev += cv::sum(tmp.mul(tmp))[0];
                }

                for (auto reg : concatenated_list[j])
                {
                    auto tmp = image(cv::Range(higth_bound[reg], botom_bound[reg]), cv::Range(left_bound[reg], right_bound[reg])) - mean_val;
                    area_std_dev += cv::sum(tmp.mul(tmp))[0];
                }

                area_std_dev = area_std_dev / (area_count - 1);

                if (area_std_dev < stddev * stddev)
                {
                    merged = true;
                    for (int t : neighbour_list[i])
                    {
                        if (t == i)
                            continue;
                        neighbour_list[t].remove(i);
                        neighbour_list[t].push_back(j);
                    }

                    neighbour_list[j].merge(neighbour_list[i]);
                    concatenated_list[j].merge(concatenated_list[i]);
                    extended_areas.insert(i);
                    break;
                }
            }
        }
    }

    cv::Mat output(image.size(), CV_8UC1);

    for (int i = 0; i < cluster_num; ++i)
    {
        if (concatenated_list[i].empty())
            continue;

        cv::Mat mean, dev;
        cv::Mat mask(image.size(), CV_8UC1);
        mask.setTo(0);

        for (auto joined_area : concatenated_list[i])
        {
            mask(cv::Range(higth_bound[joined_area], botom_bound[joined_area] + 1), cv::Range(left_bound[joined_area], right_bound[joined_area] + 1)) = 1;
        }

        cv::meanStdDev(image, mean, dev, mask);
        output.setTo(mean, mask);
    }

    return output;
}
} // namespace cvlib
