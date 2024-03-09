#include<opencv2\opencv.hpp>
#include<iostream>
#include<string>
#include<vector>

using namespace cv;
using namespace std;

//Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
void grayImageShow(cv::Mat& input, cv::Mat& output)
{
	for (int i = 0; i < input.rows; ++i)
	{
		for (int j = 0; j < input.cols; ++j)
		{
			output.at<uchar>(i, j) = cv::saturate_cast<uchar>((1140 * input.at<cv::Vec3b>(i, j)[0] + 5870 * input.at<cv::Vec3b>(i, j)[1] + 2989 * input.at<cv::Vec3b>(i, j)[2])/10000);
		}
	}
	cv::imshow("dst", output);
}

//��ʾֱ��ͼ
void showHist(Mat& img, Mat& dst, Mat& gray_Img)
{
	//1������3������������ÿ��ͨ������ͼ��ͨ����
	//�������������ͱ������洢ÿ��ͨ��������split����������ͼ�񻮷ֳ�3��ͨ����
	std::vector<Mat> bgr;
	split(img, bgr);

	//2������ֱ��ͼ��������
	int numbers = 256;

	//3�����������Χ�������������洢ÿ��ֱ��ͼ
	float range[] = { 0,256 };
	const float* histRange = { range };
	Mat b_hist, g_hist, r_hist, gray_hist;

	//4��ʹ��calcHist��������ֱ��ͼ
	int numbins = 256;
	calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbins, &histRange);
	calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbins, &histRange);
	calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbins, &histRange);
	calcHist(&gray_Img, 1, 0, Mat(), gray_hist, 1, &numbins, &histRange);

	//5������һ��512*300���ش�С�Ĳ�ɫͼ�����ڻ�����ʾ
	int width = 512;
	int height = 300;
	Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));

	//6������Сֵ�����ֵ��׼��ֱ��ͼ����
	normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
	normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
	normalize(r_hist, r_hist, 0, height, NORM_MINMAX);
	normalize(gray_hist, gray_hist, 0, height, NORM_MINMAX);

	//7��ʹ�ò�ɫͨ������ֱ��ͼ
	int binStep = cvRound((float)width / (float)numbins);  //ͨ������ȳ���������������binStep����

	for (int i = 1; i < numbins; i++)
	{
		line(histImage,
			Point(binStep * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0)
		);
		line(histImage,
			Point(binStep * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0)
		);
		line(histImage,
			Point(binStep * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255)
		);
		line(histImage,
			Point(binStep * (i - 1), height - cvRound(gray_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(gray_hist.at<float>(i))),
			Scalar(255, 255, 255)
		);
	}
	dst = histImage;
	return;
}

void onMouse(int event, int x, int y, int flags, void* param)  //evnet:����¼����� x,y:������� flags������ĸ���
{
	Mat* im = reinterpret_cast<Mat*>(param);
	switch (event) {

	case EVENT_LBUTTONDOWN:
		//��ʾͼ������ֵ

		if (static_cast<int>(im->channels()) == 1)
		{
			//��ͼ��Ϊ��ͨ��ͼ������ʾ������������Լ��Ҷ�ֵ
			switch (im->type())
			{
			case 0:
				cout << "at (" << x << ", " << y << " ) value is: " << static_cast<int>(im->at<uchar>(Point(x, y))) << endl; break;
			case 1:
				cout << "at (" << x << ", " << y << " ) value is: " << static_cast<int>(im->at<char>(Point(x, y))) << endl; break;
			case 2:
				cout << "at (" << x << ", " << y << " ) value is: " << static_cast<int>(im->at<ushort>(Point(x, y))) << endl; break;
			case 3:
				cout << "at (" << x << ", " << y << " ) value is: " << static_cast<int>(im->at<short>(Point(x, y))) << endl; break;
			case 4:
				cout << "at (" << x << ", " << y << " ) value is: " << static_cast<int>(im->at<int>(Point(x, y))) << endl; break;
			case 5:
				cout << "at (" << x << ", " << y << " ) value is: " << static_cast<int>(im->at<float>(Point(x, y))) << endl; break;
			case 6:
				cout << "at (" << x << ", " << y << " ) value is: " << static_cast<int>(im->at<double>(Point(x, y))) << endl; break;
			}
		}
		else
		{
			//��ͼ��Ϊ��ɫͼ������ʾ����������Լ���Ӧ��B, G, Rֵ
			int B = static_cast<int>(im->at<Vec3b>(Point(x, y))[0]);
			int G = static_cast<int>(im->at<Vec3b>(Point(x, y))[1]);
			int R = static_cast<int>(im->at<Vec3b>(Point(x, y))[2]);
			cout << R << " " << G << " " << B << endl;
		}
		break;
	}
}

Mat calcGrayHist(const Mat& image)
{
	Mat histogram = Mat::zeros(Size(256, 1), CV_32SC1);
	int rows = image.rows;
	int cols = image.cols;
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int index = int(image.at<uchar>(r, c));
			histogram.at<int>(0, index) += 1;
		}
	}
	return histogram;
}

int threshTwoPeaks(const Mat& image, Mat& thresh_out)
{
	// ����Ҷ�ֱ��ͼ
	Mat histogram = calcGrayHist(image);
	// �ҵ��Ҷ�ֱ��ͼ����ֵ��Ӧ�ĻҶ�ֵ
	Point firstPeakLoc;
	minMaxLoc(histogram, NULL, NULL, NULL, &firstPeakLoc);
	int firstPeak = firstPeakLoc.x;
	//Ѱ�һҶ�ֱ��ͼ�ĵڶ�����ֵ��Ӧ�ĻҶ�ֵ
	Mat measureDists = Mat::zeros(Size(256, 1), CV_32FC1);
	for (int k = 0; k < 256; k++)
	{
		int hist_k = histogram.at<int>(0, k);
		measureDists.at<float>(0, k) = pow(float(k - firstPeak), 2) * hist_k;
	}
	Point secondPeakLoc;
	minMaxLoc(measureDists, NULL, NULL, NULL, &secondPeakLoc);
	int secondPeak = secondPeakLoc.x;
	//�ҵ�������ֵ֮�����Сֵ��Ӧ�ĻҶ�ֵ����Ϊ��ֵ
	Point threshLoc;
	int thresh = 0;
	if (firstPeak < secondPeak) {
		minMaxLoc(histogram.colRange(firstPeak, secondPeak), NULL, NULL, &threshLoc);
		thresh = firstPeak + threshLoc.x + 1;
	}
	else {
		minMaxLoc(histogram.colRange(secondPeak, firstPeak), NULL, NULL, &threshLoc);
		thresh = secondPeak + threshLoc.x + 1;
	}
	//��ֵ�ָ�
	threshold(image, thresh_out, thresh, 255, THRESH_BINARY);
	return thresh;
}

int main()
{
	//��ɫת�Ҷ�
	Mat src, grey;
	//Mat dst;
	src=imread("llw.jpg");
	if(src.empty())
	{
		printf("����ʧ��\n");
		return -1;
	}

	//��ȡ����ֵ
	namedWindow("src", WINDOW_NORMAL);
	imshow("src", src);
	setMouseCallback("src", onMouse, reinterpret_cast<void*>(&src));

	cvtColor(src,grey,COLOR_BGR2GRAY);
	//dst.create(grey.rows, grey.cols, CV_8UC1);
	//grayImageShow(src, dst);
	//namedWindow("src", WINDOW_NORMAL);
	//namedWindow("grey", WINDOW_NORMAL);
	//namedWindow("dst", WINDOW_NORMAL);
	//imshow("src", src);
	//imshow("grey",grey);
	//imshow("dst", dst);


	Mat channel[3];
	split(src, channel);
	//namedWindow("B", WINDOW_NORMAL);
	//imshow("B", channel[0]);
	//namedWindow("G", WINDOW_NORMAL);
	//imshow("G", channel[1]);
	//namedWindow("R", WINDOW_NORMAL);
	//imshow("R", channel[2]);

	////grayImageShow(src, dst);
	//Mat histImage;
	//showHist(src, histImage,grey);
	//namedWindow("histImage", 0);
	//imshow("histImage", histImage);


	////���� ������ֵ��
	//Mat threshImage_grey, threshImage_R, threshImage_G, threshImage_B;
	//int thresh_grey = threshTwoPeaks(grey, threshImage_grey);
	//int thresh_R = threshTwoPeaks(channel[2], threshImage_R);
	//int thresh_G = threshTwoPeaks(channel[1], threshImage_G);
	//int thresh_B = threshTwoPeaks(channel[0], threshImage_B);
	//cout << "threshold value grey:" << thresh_grey << endl;
	//cout << "threshold value R:" << thresh_R << endl;
	//cout << "threshold value G:" << thresh_G << endl;
	//cout << "threshold value B:" << thresh_B << endl;
	////��ʾ��ֵ��Ķ�ֵͼ
	//namedWindow("threshold grey", WINDOW_NORMAL);
	//imshow("threshold grey", threshImage_grey);
	//namedWindow("threshold R", WINDOW_NORMAL);
	//imshow("threshold R", threshImage_R);
	//namedWindow("threshold G", WINDOW_NORMAL);
	//imshow("threshold G", threshImage_G);
	//namedWindow("threshold B", WINDOW_NORMAL);
	//imshow("threshold B", threshImage_B);


	////��ֵ��ȡ
	//Mat thredG, thredB, thredR;
	////��ֵ�˲�
	//blur(grey, grey, Size(3, 3));
	//threshold(channel[0], thredG, 100, 255, THRESH_BINARY);//binary
	//threshold(channel[1], thredB, 120, 255, THRESH_BINARY);//binary
	//threshold(channel[2], thredR, 98, 255, THRESH_BINARY);//binary
	////threshold(grey, thred, 200, 255, THRESH_BINARY);//OTSU
	////adaptiveThreshold(grey, thredGray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 8, 3);//����Ӧ��ֵ
	//namedWindow("thredG", WINDOW_NORMAL);
	//imshow("thredG", thredG);
	//namedWindow("THRESH", WINDOW_NORMAL);
	//imshow("thredB", thredB);
	//namedWindow("thredR", WINDOW_NORMAL);
	//imshow("thredR", thredR);



	waitKey(0);
	return 0;
}