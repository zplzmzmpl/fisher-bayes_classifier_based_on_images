#include "Histogram1D.h"

Histogram1D::Histogram1D()
{
    histSize[0] = 256;//���Ӹ�����Ϊ256
    channels[0] = 0;//ʹ��һ��ͨ����Ĭ��Ϊ0
    hranges[0] = 0.0;
    hranges[1] = 256.0;
    ranges[0] = hranges;//ֵ��Χ
}

Histogram1D::~Histogram1D()
{
}

cv::Mat Histogram1D::getHistogram(const cv::Mat& image)
{
    cv::Mat hist;
    cv::calcHist(&image,
        1,//����ͼ���ֱ��ͼ
        channels,
        cv::Mat(),
        hist,
        1,//һάֱ��ͼ
        histSize,
        ranges);
    return hist;
}

cv::Mat getImageOfHistogram(const cv::Mat& hist,
    int zoom//��������ϵ��
)
{
    double minVal = 0;
    double maxVal = 0;
    cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

    int histSize = hist.rows;

    //������ʾֱ��ͼ�ķ���ͼ��
    cv::Mat histImg(histSize * zoom, histSize * zoom, CV_8U, cv::Scalar(255));

    //������ߵ�Ϊ90%�����Ӹ���
    int hmax = static_cast<int>(histSize * 0.9);

    //����
    for (int h = 0; h < histSize; h++)
    {
        float binVal = hist.at<float>(h);
        if (binVal > 0)
        {
            int intensity = binVal * hmax / maxVal;
            cv::line(histImg, cv::Point(h * zoom, histSize * zoom), cv::Point(h * zoom, (histSize - intensity) * zoom), cv::Scalar(0), zoom);
        }
    }
    return histImg;
}

cv::Mat Histogram1D::getHistogramImage(const cv::Mat& image, int zoom = 1)
{
    cv::Mat hist = getHistogram(image);
    //������þ�̬����
    return getImageOfHistogram(hist, zoom);
}
//Ӧ�ò��ұ���صĺ���
cv::Mat Histogram1D::applyLookUp(const cv::Mat& image, const cv::Mat& lookup)
{
    cv::Mat result;
    cv::LUT(image, lookup, result);
    return result;
}
//get��������ȡ��Ա����histSize
int* Histogram1D::getHistSize()
{
    return histSize;
}
//set���������ó�Ա����histSize
void  Histogram1D::setHistSize(int n)
{
    histSize[0] = n;
}