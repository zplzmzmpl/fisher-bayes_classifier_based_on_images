#pragma once
#include"opencv2/opencv.hpp"
#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"

class Histogram1D
{
private:
    int channels[1];//ʹ�õ�ͨ������
    int histSize[1];//ֱ��ͼ���ӣ�bin��������
    const float* ranges[1];//����ֵ��Χ
    float hranges[2];
public:
    Histogram1D();
    cv::Mat getHistogram(const cv::Mat& image);//�õ�ֱ��ͼ
    void setHistSize(int n);//����ֱ��ͼ��������
    int* getHistSize();
    static cv::Mat applyLookUp(const cv::Mat& image, const cv::Mat& lookup);//Ӧ�ò��ұ�
    cv::Mat getHistogramImage(const cv::Mat& image, int zoom);//�õ�ֱ��ͼͼ��
    ~Histogram1D();
};