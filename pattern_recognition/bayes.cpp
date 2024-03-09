//基本步骤：
//1、输入类数, 特征数, 待分样本数
//2、输入训练样本数和训练样本集
//3、计算先验概率
//4、计算各类条件概率密度
//5、计算各类的后验概率
//6.1、若按最小错误率原则分类, 则根据后验概率判定
//6.2、若按最小风险原则分类, 则计算各样本属于各类时的风险并判定
#include<iostream>
#include<fstream>
#include<sstream>
#include<cmath>
#include<cstdio>
#include<opencv2/opencv.hpp>
using namespace std;

const int MAXN = 100;
const double pi = 3.1415926;

ifstream cin1("llw-apple.txt");
ifstream cin2("llw-banana.txt");
ifstream cin3("llw-background.txt");
ifstream cin4("test.txt");
ofstream cout1("result.txt");

struct FRUIT
{
    int R;
    int G;
    int B;
};

FRUIT apple[MAXN];
FRUIT orange[MAXN];
FRUIT background[MAXN];
int apple_num;
int orange_num;
int bk_num;

double P_apple;
double P_orange;
double P_bk;

struct NORMAL
{
    double mu1;
    double mu2;
    double mu3;
    double delta1;
    double delta2;
    double delta3;
    double rho12;
    double rho13;
    double rho23;
};

NORMAL apple_normal;
NORMAL orange_normal;
NORMAL bk_normal;

void In()
{
    apple_num = 0;
    orange_num = 0;
    while (cin1 >> apple[apple_num + 1].R >> apple[apple_num + 1].G >> apple[apple_num + 1].B)
    {
        apple_num++;
    }
    cout << "amounts of apple samples: " << apple_num << endl;
    while (cin2 >> orange[orange_num + 1].R >> orange[orange_num + 1].G >> orange[orange_num + 1].B)
    {
        orange_num++;
    }
    cout << "amounts of orange samples: " << orange_num << endl;
    while (cin3 >> background[bk_num + 1].R >> background[bk_num + 1].G >> background[bk_num + 1].B)
    {
        bk_num++;
    }
    cout << "amounts of background samples: " << bk_num << endl;

}
void Init()
{
    double total = apple_num + orange_num + bk_num;
    P_apple = ((double)apple_num / total);
    P_orange = ((double)orange_num / total);
    P_bk = ((double)bk_num / total);
    cout << "prior passiblity of apple: " << P_apple << endl;
    cout << "prior passiblity of orange: " << P_orange << endl;
    cout << "prior passiblity of background: " << P_bk << endl;
}


void Normalization(struct FRUIT* fruit, int fruit_num, struct NORMAL& fruit_normal)
{
    double mu1 = 0, mu2 = 0, mu3 = 0;
    double delta1 = 0, delta2 = 0, delta3 = 0;
    double rho12 = 0, rho13 = 0, rho23 = 0;

    for (int i = 1; i <= fruit_num; i++)
    {
        mu1 += fruit[i].R;
        mu2 += fruit[i].G;
        mu3 += fruit[i].B;
    }

    mu1 /= fruit_num;
    mu2 /= fruit_num;
    mu3 /= fruit_num;

    for (int i = 1; i <= fruit_num; i++)
    {
        delta1 += (fruit[i].R - mu1) * (fruit[i].R - mu1);
        delta2 += (fruit[i].G - mu2) * (fruit[i].G - mu2);
        delta3 += (fruit[i].B - mu3) * (fruit[i].B - mu3);
    }

    delta1 /= fruit_num;
    delta2 /= fruit_num;
    delta3 /= fruit_num;

    delta1 = sqrt(delta1);
    delta2 = sqrt(delta2);
    delta3 = sqrt(delta3);

    for (int i = 1; i <= fruit_num; i++)
    {
        rho12 += (fruit[i].R - mu1) * (fruit[i].G - mu2);
        rho13 += (fruit[i].R - mu1) * (fruit[i].B - mu3);
        rho23 += (fruit[i].G - mu2) * (fruit[i].B - mu3);
    }

    rho12 /= fruit_num;
    rho12 /= (delta1 * delta2);
    rho13 /= fruit_num;
    rho13 /= (delta1 * delta3);
    rho23 /= fruit_num;
    rho23 /= (delta2 * delta3);

    fruit_normal.mu1 = mu1;
    fruit_normal.mu2 = mu2;
    fruit_normal.mu3 = mu3;
    fruit_normal.delta1 = delta1;
    fruit_normal.delta2 = delta2;
    fruit_normal.delta3 = delta3;
    fruit_normal.rho12 = rho12;
    fruit_normal.rho13 = rho13;
    fruit_normal.rho23 = rho23;

    cout << mu1 << " " << delta1 << " " << mu2 << " " << delta2 << " " << mu3 << " " << delta3 << " " << rho12 << " " << rho13 << " " << rho23 << endl;
}

double P(NORMAL& normal, double x1, double x2, double x3)
{
    double ans;
    double mu1 = normal.mu1;
    double mu2 = normal.mu2;
    double mu3 = normal.mu3;
    double delta1 = normal.delta1;
    double delta2 = normal.delta2;
    double delta3 = normal.delta3;
    double rho12 = normal.rho12;
    double rho13 = normal.rho13;
    double rho23 = normal.rho23;

    ans = (1 / (pow((2 * pi), 1.5) * delta1 * delta2 * delta3)) * exp(-0.5 * (pow((x1 - mu1) / delta1, 2) + pow((x2 - mu2) / delta2, 2) + pow((x3 - mu3) / delta3, 2)));
    return ans;
}

double Posterior_probability1(double x1, double x2, double x3, int t)
{
    double Pos_apple = P(apple_normal, x1, x2, x3) * P_apple;
    double Pos_orange = P(orange_normal, x1, x2, x3) * P_orange;
    double Pos_bk = P(bk_normal, x1, x2, x3) * P_bk;
    double Pos_sum = Pos_apple + Pos_orange + Pos_bk;
    if (t == 0)
        return Pos_apple / Pos_sum;
    else if (t == 1)
        return Pos_orange / Pos_sum;
    else if (t == 2)
        return Pos_bk / Pos_sum;
}

int Classify(double x1, double x2, double x3)
{
    double apple_pos = Posterior_probability1(x1, x2, x3, 0);
    double orange_pos = Posterior_probability1(x1, x2, x3, 1);
    double bk_pos = Posterior_probability1(x1, x2, x3, 2);

    if (apple_pos >= orange_pos && apple_pos >= bk_pos) return 0;
    else if (orange_pos >= apple_pos && orange_pos >= bk_pos) return 1;
    else return 2;
}

vector<int> classifyPixels(const vector<vector<int>>& rgbValues) {
    vector<int> classifications;

    for (const auto& rgb : rgbValues) {
        double r = rgb[0];  // Red value
        double g = rgb[1];  // Green value
        double b = rgb[2];  // Blue value

        int classification = Classify(r, g, b);
        classifications.push_back(classification);
    }

    return classifications;
}

void Find_error_rate()
{
    FRUIT f;
    char c;
    int right_num = 0;
    int wrong_num = 0;
    string line;
    while (getline(cin4,line))
    {
        istringstream iss(line);
        iss >> f.R >> f.G >> f.B >> c;
        //cout << f.R << " " << f.G << " " << f.B << endl;
        int judge = Classify(f.R, f.G, f.B);
        if ((c == 'a' || c == 'A') && judge == 0)
            right_num++;
        else if ((c == 'o' || c == 'O') && judge == 1)
            right_num++;
        else if ((c == 'b' || c == 'B') && judge == 2)
            right_num++;
        else
            wrong_num++;

        cout1 << f.R << " " << f.G << " " << f.B << " " << judge << endl;
    }
    cout1 << "error rate is " << (double)wrong_num / (double)(wrong_num + right_num) << endl;
    cout << "error rate is " << (double)wrong_num / (double)(wrong_num + right_num) << endl;
}

vector<vector<int>> getRGBValues(const cv::Mat& image) {
    vector<vector<int>> rgbValues;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            vector<int> rgb;
            rgb.push_back(pixel[2]);  // Red value
            rgb.push_back(pixel[1]);  // Green value
            rgb.push_back(pixel[0]);  // Blue value
            rgbValues.push_back(rgb);
        }
    }

    return rgbValues;
}

cv::Mat convertToNewImage(const vector<int>& values, int rows, int cols) {
    cv::Mat newImage(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int value = values[i * cols + j];
            cv::Vec3b& pixel = newImage.at<cv::Vec3b>(i, j);

            if (value == 0) {
                pixel[0] = 48;
                pixel[1] = 148;
                pixel[2] = 250;
            }
            else if (value == 1) {
                pixel[0] = 148;
                pixel[1] = 148;
                pixel[2] = 50;
            }
            else if (value == 2) {
                pixel[0] = 48;
                pixel[1] = 138;
                pixel[2] = 34;
            }
        }
    }

    return newImage;
}

int main()
{

    //-----------------single pixel classification-----------------
    In();
    Init();
    //normalization features
    Normalization(apple, apple_num, apple_normal);
    Normalization(orange, orange_num, orange_normal);
    Normalization(background, bk_num, bk_normal);
    //classify single pixel and find error rate
    Find_error_rate();


    //-----------------whole image classification-------------------
    // load the image
    cv::Mat image = cv::imread("llw.jpg");

    // check if the image was loaded successfully
    if (image.empty()) {
        cout << "Failed to load the image." << endl;
        return -1;
    }

    // get the RGB values of the image
    vector<vector<int>> rgbValues = getRGBValues(image);

    //processing all values as single classification
    //restore result as a vector<int> full of (0 or 1 or 2)
    vector<int> result = classifyPixels(rgbValues);

    //convert result to result
    cv::Mat cls = convertToNewImage(result, image.rows, image.cols);

    //restore as image
    cv::imwrite("bayes-llw.jpg", cls);
    //cv::imshow("result", cls);


    system("pause");
    return 0;
}