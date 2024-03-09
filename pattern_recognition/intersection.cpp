//利用像片外方位元素进行前方交会
#include<iostream>
#include<fstream>
#include <iomanip>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	//读取同名像点坐标到数组
	/*double T_Z_B[2][2] = { {0.051758,	0.080555},
		{-0.039953,	0.078463} 
	};*/
	double samepoint[5][4] = {
	{0.051758, 0.080555, -0.039953, 0.078463},
	{0.014618, -0.00231, -0.076006, 0.00036},
	{0.04988, -0.00782, -0.042201, -0.01022},
	{0.08614, -0.01346, -0.07706, -0.02112},
	{0.048035, -0.079962, -0.044438, -0.079736}
	};
	int i = 4;
	double x1 = samepoint[i][0];
	double y1 = samepoint[i][1];
	double x2 = samepoint[i][2];
	double y2 = samepoint[i][3];
	//定义内方位元素
	double x0 = 0.00000;//mm
	double y0 = 0.00000;//mm
	double f = 150;//mm
	//从文件中读取外方位元素到数组
	/*int i, j;*/
	double data[2][6] = {
		{40952493.794029  ,   33982061.872285  ,   129306613.690970, -159.887771 ,    49.576077 ,  90.173915},
		{-6532174.913475  ,   72840541.194954 ,    80706368.324398, -216.878884  ,   104.759882, -21.406311}
	};
	
	//double rotate1[3][3]{
	//	{0.396006, 0.882832, 0.252560},
	//	{ 0.619596, -0.459897, 0.636078},
	//	{ 0.677702, 0.095406, -0.729121}
	//};
	//double rotate3[3][3]{
	//	{0.775686, -0.629090,0.050565},
	//	{ 0.256630, 0.387598,0.885386 },
	//	{-0.576587, -0.673805,0.462097} 
	//};

	//左右像片的外方位元素
	double Xs1 = data[0][0];
	double Ys1 = data[0][1];
	double Zs1 = data[0][2];
	double phi1 = data[0][3];
	double omig1 = data[0][4];
	double kappa1 = data[0][5];
	double Xs2 = data[1][0];
	double Ys2 = data[1][1];
	double Zs2 = data[1][2];
	double phi2 = data[1][3];
	double omig2 = data[1][4];
	double kappa2 = data[1][5];
	//cout.precision(12);//控制输出的小数点位数
	//计算摄影基线的三个分量
	double Bx = Xs2 - Xs1;
	double By = Ys2 - Ys1;
	double Bz = Zs2 - Zs1;
	//利用外方位角元素计算左右像片的旋转矩阵R1和R2，用OpenCV矩阵，方便
	double a1 = cos(phi1) * cos(kappa1) - sin(phi1) * sin(omig1) * sin(kappa1);
	double a2 = -cos(phi1) * sin(kappa1) - sin(phi1) * sin(omig1) * cos(kappa1);
	double a3 = -sin(phi1) * cos(omig1);
	double b1 = cos(omig1) * sin(kappa1);
	double b2 = cos(omig1) * cos(kappa1);
	double b3 = -sin(omig1);
	double c1 = sin(phi1) * cos(kappa1) + cos(phi1) * sin(omig1) * sin(kappa1);
	double c2 = -sin(phi1) * sin(kappa1) + cos(phi1) * sin(omig1) * cos(kappa1);
	double c3 = cos(phi1) * cos(omig1);
	Mat R1 = Mat::ones(3, 3, CV_64F);
	R1.at<double>(0, 0) = a1;
	R1.at<double>(0, 1) = a2;
	R1.at<double>(0, 2) = a3;
	R1.at<double>(1, 0) = b1;
	R1.at<double>(1, 1) = b2;
	R1.at<double>(1, 2) = b3;
	R1.at<double>(2, 0) = c1;
	R1.at<double>(2, 1) = c2;
	R1.at<double>(2, 2) = c3;

	double a11 = cos(phi2) * cos(kappa2) - sin(phi2) * sin(omig2) * sin(kappa2);
	double a22 = -cos(phi2) * sin(kappa2) - sin(phi2) * sin(omig2) * cos(kappa2);
	double a33 = -sin(phi2) * cos(omig2);
	double b11 = cos(omig2) * sin(kappa2);
	double b22 = cos(omig2) * cos(kappa2);
	double b33 = -sin(omig2);
	double c11 = sin(phi2) * cos(kappa2) + cos(phi2) * sin(omig2) * sin(kappa2);
	double c22 = -sin(phi2) * sin(kappa2) + cos(phi2) * sin(omig2) * cos(kappa2);
	double c33 = cos(phi2) * cos(omig2);
	Mat R2 = Mat::ones(3, 3, CV_64F);
	R2.at<double>(0, 0) = a11;
	R2.at<double>(0, 1) = a22;
	R2.at<double>(0, 2) = a33;
	R2.at<double>(1, 0) = b11;
	R2.at<double>(1, 1) = b22;
	R2.at<double>(1, 2) = b33;
	R2.at<double>(2, 0) = c11;
	R2.at<double>(2, 1) = c22;
	R2.at<double>(2, 2) = c33;

	//计算同名像点的像空间辅助坐标系(X1,Y1,Z1)与(X2,Y2,Z2)
	Mat RR1, RR2;
	Mat R3 = Mat::ones(3, 1, CV_64F);
	R3.at<double>(0, 0) = x1;
	R3.at<double>(1, 0) = y1;
	R3.at<double>(2, 0) = -f;
	Mat R33 = Mat::ones(3, 1, CV_64F);
	R33.at<double>(0, 0) = x2;
	R33.at<double>(1, 0) = y2;
	R33.at<double>(2, 0) = -f;
	RR1 = R1 * R3;
	double X1 = RR1.at<double>(0, 0);
	double Y1 = RR1.at<double>(1, 0);
	double Z1 = RR1.at<double>(2, 0);

	RR2 = R2 * R33;
	double X2 = RR2.at<double>(0, 0);
	double Y2 = RR2.at<double>(1, 0);
	double Z2 = RR2.at<double>(2, 0);

	//计算投影系数N1,N2
	double N1 = (Bx * Z2 - Bz * X2) / (X1 * Z2 - X2 * Z1);
	double N2 = (Bx * Z1 - Bz * X1) / (X1 * Z2 - X2 * Z1);

	//计算地面点的左像辅系坐标(deteX,deteY,deteZ)
	double deteX = N1 * X1;
	double deteY = 0.5 * (N1 * Y1 + N2 * Y2 + By);
	double deteZ = N1 * Z1;
	//计算地面点的地面坐标(X,Y,Z)
	double X = Xs1 + deteX;
	double Y = Ys1 + deteY;
	double Z = Zs1 + deteZ;
	cout.precision(11);
	cout << "计算得出地面点的地面坐标分别是：" << endl
		<< "X=" << X << endl
		<< "Y=" << Y << endl
		<< "Z=" << Z << endl;
	cout << "You have finished the work  !" << endl;
	system("pause");
	return 0;
	//point group1
	//X = 9048141.5654
	//Y = 148090517.18
	//Z = 221658553.76

	//point group2
	//X = 9008856.2728
	//Y = 148213502.94
	//Z = 221543237.33

	//point group3
	//X = 8959584.2504
	//Y = 148300759.82
	//Z = 221728337.9

	//point group4
	//X = 8999097.1987
	//Y = 148350885
	//Z = 221658253.71

	//point group5
	//X = 8890086.5716
	//Y = 148463622.17
	//Z = 221776578.35
}
