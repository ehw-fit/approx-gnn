#include <opencv2/opencv.hpp>

double getPSNR(const cv::Mat& I1, const cv::Mat& I2);
cv::Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2);