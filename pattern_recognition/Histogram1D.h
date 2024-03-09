#pragma once
#include"opencv2/opencv.hpp"
#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"

class Histogram1D
{
private:
    int channels[1];//使用的通道数量
    int histSize[1];//直方图箱子（bin）的数量
    const float* ranges[1];//像素值范围
    float hranges[2];
public:
    Histogram1D();
    cv::Mat getHistogram(const cv::Mat& image);//得到直方图
    void setHistSize(int n);//设置直方图箱子数量
    int* getHistSize();
    static cv::Mat applyLookUp(const cv::Mat& image, const cv::Mat& lookup);//应用查找表
    cv::Mat getHistogramImage(const cv::Mat& image, int zoom);//得到直方图图像
    ~Histogram1D();
};