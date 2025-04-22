#include <iostream>
#include <functional>
#include <vector>
#include <string>
#include <array>
#include <filesystem>
#include <numeric>
#include <sstream>

#include "configs.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>

uint8_t getMatData(const cv::Mat &m, int x, int y)
{
    if (y < 0)
        y = 0;
    if (y >= m.rows)
        y = m.rows - 1;

    if (x < 0)
        x = 0;
    if (x >= m.cols)
        x = m.cols - 1;
    return m.at<uint8_t>(y, x);
}

static uint8_t saturate(int m)
{
    if (m < 0)
        return 0;
    if (m > 255)
        return 255;
    return m;
}

void setMatData(cv::Mat &m, int x, int y, int val)
{
    if (y < 0)
        y = 0;
    if (y >= m.rows)
        y = m.rows - 1;

    if (x < 0)
        x = 0;
    if (x >= m.cols)
        x = m.cols - 1;

    m.at<uint8_t>(y, x) = saturate(val);
}

void process_config(const std::vector<cv::Mat> &images, tWiring config, std::vector<tAdd> &adders, std::vector<cv::Mat> &results, int divisor)
{
    int i = 0;

    for (const auto &image : images)
    {
        for (int y = 0; y < image.rows; y++)
        {
            for (int x = 0; x < image.cols; x++)
            {
                auto val = config(
                    getMatData(image, x - 1, y - 1),
                    getMatData(image, x, y - 1),
                    getMatData(image, x + 1, y - 1),
                    getMatData(image, x - 1, y),
                    getMatData(image, x, y),
                    getMatData(image, x + 1, y),
                    getMatData(image, x - 1, y + 1),
                    getMatData(image, x, y + 1),
                    getMatData(image, x + 1, y + 1),
                    adders.data());

                setMatData(results[i], x, y, val / divisor);
            }
        }
        i++;
    }
}

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " image1 [image2 image3 ...]" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string skip_to = "";
    char **imgNames;
    int imgCount;
    if (strcmp(argv[1], "--from") == 0) {
        skip_to = argv[2];
        imgNames = argv + 3;
        imgCount = argc - 3;
    }
    else {
        imgNames = argv + 1;
        imgCount = argc - 1;
    }
    bool skip_configs = skip_to != "";

    std::vector<cv::Mat> inputImages(imgCount);

    for (int imgid = 0; imgid < imgCount; imgid++)
    {
        auto &A = inputImages[imgid];
        if (!std::filesystem::exists(imgNames[imgid]))
        {
            std::cerr << "Evaluation image file " << imgNames[imgid] << " does not exist.";
            exit(EXIT_FAILURE);
        }

        A = cv::imread(imgNames[imgid], cv::IMREAD_COLOR);
        cv::cvtColor(A, A, CV_8U);
        cv::cvtColor(A, A, cv::COLOR_BGR2GRAY);
    }

    std::vector<cv::Mat> result_images;
    std::vector<cv::Mat> reference_images;
    result_images.resize(inputImages.size());
    reference_images.resize(inputImages.size());
    for (int i = 0; i < inputImages.size(); i++)
    {
        result_images[i].create(inputImages[i].rows, inputImages[i].cols, CV_8U);
        reference_images[i].create(inputImages[i].rows, inputImages[i].cols, CV_8U);
    }

    auto app_config = Configs();
    app_config.load();

    if (!skip_configs) {
        std::cout << "wiring,config,psnr,ssim" << std::endl;
    }

    std::string config_name;
    std::string wiring_name;
    std::string last_wiring_name;
    std::string line;
    std::string word;
    int adder_count = 0;
    int divisor = 1;

    std::vector<tAdd> adders{};

    while (std::cin >> config_name)
    {
        if (config_name == "")
        {
            break;
        }

        std::cin >> wiring_name >> adder_count >> divisor;

        if (wiring_name == skip_to) {
            skip_configs = false;
        }

        if (skip_configs) {
            continue;
        }

        if (wiring_name != last_wiring_name) {
            adders.clear();

            for (int i = 0; i < adder_count; i++)
            {
                adders.push_back(app_config.adders["add_ref"]);
            }

            process_config(
                inputImages,
                app_config.wirings[wiring_name],
                adders,
                reference_images, divisor);

            last_wiring_name = wiring_name;
        }

        adders.clear();

        for (int i = 0; i < adder_count; i++)
        {
            std::cin >> word;
            adders.push_back(app_config.adders[word]);
        }

        process_config(
            inputImages,
            app_config.wirings[wiring_name],
            adders,
            result_images, divisor);


        std::vector<float> psnrs(inputImages.size());
        std::vector<float> ssims(inputImages.size());

        for (size_t i = 0; i < inputImages.size(); i++)
        {
            psnrs[i] = getPSNR(reference_images[i], result_images[i]);
            ssims[i] = getMSSIM(reference_images[i], result_images[i])[0];
        }

        float avg_psnr = std::accumulate(psnrs.begin(), psnrs.end(), 0.0f) / (float)psnrs.size();
        float avg_ssim = std::accumulate(ssims.begin(), ssims.end(), 0.0f) / (float)ssims.size();

        std::cout << wiring_name << "," << config_name << "," << avg_psnr << "," << avg_ssim << std::endl;
    }
}
