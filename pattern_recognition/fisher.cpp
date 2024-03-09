#include <iostream>
#include<fstream>
#include <Eigen/LU>
#include <vector>
#include<opencv2/opencv.hpp>

using namespace Eigen;
void load_data(std::ifstream& file, std::vector<std::vector<double>>& data)
{
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double val;
            std::vector<double> row;
            while (iss >> val) {
                //std::cout << val << std::endl;
                row.push_back(val);
            }
            data.push_back(row);
        }
    }
    else {
        std::cout << "open faied" << std::endl;
    }
}

std::vector<std::vector<int>> getRGBValues(const cv::Mat& image) {
    std::vector<std::vector<int>> rgbValues;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            std::vector<int> rgb;
            rgb.push_back(pixel[2]);  // Red value
            rgb.push_back(pixel[1]);  // Green value
            rgb.push_back(pixel[0]);  // Blue value
            rgbValues.push_back(rgb);
        }
    }

    return rgbValues;
}

cv::Mat convertToNewImage(const std::vector<int>& values, int rows, int cols) {
    cv::Mat newImage(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    std::cout << "begin writing..." << std::endl;
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
            else
            {
                pixel[0] = 255;
                pixel[1] = 255;
                pixel[2] = 255;
            }
        }
    }
    std::cout << "Done" << std::endl;
    return newImage;
}

int main() {
    std::string path = "data-llw.txt"; // Replace with your file path
    std::ifstream file(path);
    std::vector<std::vector<double>> data;
    load_data(file, data);
    int numSamples = data.size();
    int numFeatures = data[0].size();

    int numRows1 = 41;
    int numRows2 = 46;

    MatrixXd df(numSamples, numFeatures);
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            df(i, j) = data[i][j];
        }
    }
    MatrixXd df1 = df.block(0, 0, numRows1, numFeatures);
    MatrixXd df2 = df.block(numRows1, 0, numRows2, numFeatures);
    MatrixXd df3 = df.block(numRows1 + numRows2, 0, numSamples - numRows1 - numRows2, numFeatures);

    // Class mean vectors
    VectorXd m1 = df1.colwise().mean();
    VectorXd m2 = df2.colwise().mean();
    VectorXd m3 = df3.colwise().mean();

    //std::cout << "各样本均值向量为：" << std::endl;
    //std::cout << "m1:\n" << m1 << std::endl;
    //std::cout << "m2:\n" << m2 << std::endl;
    //std::cout << "m3:\n" << m3 << std::endl;

    // Class scatter matrices
    MatrixXd s1 = MatrixXd::Zero(3, 3);
    MatrixXd s2 = MatrixXd::Zero(3, 3);
    MatrixXd s3 = MatrixXd::Zero(3, 3);

    for (int i = 0; i < df1.rows(); i++) {
        VectorXd a = df1.row(i).transpose() - m1;
        s1 += a * a.transpose();
    }
    for (int i = 0; i < df2.rows(); i++) {
        VectorXd c = df2.row(i).transpose() - m2;
        s2 += c * c.transpose();
    }
    for (int i = 0; i < df3.rows(); i++) {
        VectorXd e = df3.row(i).transpose() - m3;
        s3 += e * e.transpose();
    }

    //std::cout<<"样本类内离散度矩阵为：" << std::endl;
    //std::cout << "s1:\n" << s1 << std::endl;
    //std::cout << "s2:\n" << s2 << std::endl;
    //std::cout << "s3:\n" << s3 << std::endl;

    // Total within-class scatter matrices
    MatrixXd sw12 = s1 + s2;
    MatrixXd sw13 = s1 + s3;
    MatrixXd sw23 = s2 + s3;

    // Projection directions
    VectorXd a = m1 - m2;
    MatrixXd w12 = (sw12.inverse() * a).transpose();
    VectorXd b = m1 - m3;
    MatrixXd w13 = (sw13.inverse() * b).transpose();
    VectorXd c = m2 - m3;
    MatrixXd w23 = (sw23.inverse() * c).transpose();

    //std::cout << "总体类内离散度矩阵为：" << std::endl;
    //std::cout << "sw12:\n" << sw12 << std::endl;
    //std::cout << "sw13:\n" << sw13 << std::endl;
    //std::cout << "sw23:\n" << sw23 << std::endl;

    // 判别函数以及阈值T（即w0）
    double T12 = -0.5 * (m1 + m2).dot(sw12.inverse() * a);
    double T13 = -0.5 * (m1 + m3).dot(sw13.inverse() * b);
    double T23 = -0.5 * (m2 + m3).dot(sw23.inverse() * c);

    // 加载图片
    cv::Mat image = cv::imread("llw.jpg");

    // 检查是否加载成功
    if (image.empty()) {
        std::cout << "Failed to load the image." << std::endl;
        return -1;
    }

    // 获取RGB
    std::vector<std::vector<int>> rgbValues = getRGBValues(image);

    int samples = rgbValues.size();
    int features = rgbValues[0].size();
    std::cout << "samples: " << samples << " features: " << features << std::endl;
    MatrixXd test(samples, features);
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < features; j++) {
            test(i, j) = rgbValues[i][j];
        }
    }

    std::cout << "load test file success" << std::endl;

    std::vector<int> result;

    for (int i = 0; i < test.rows(); i++) {
        VectorXd x = test.row(i).transpose();
        double g12 = (w12 * x)(0) + T12;
        double g13 = (w13 * x)(0) + T13;
        double g23 = (w23 * x)(0) + T23;
        if (g12 > 0 && g13 > 0) {
            result.push_back(0);
        }
        else if (g12 < 0 && g23 > 0) {
            result.push_back(1);
        }
        else if (g13 < 0 && g23 < 0) {
            result.push_back(2);
        }
        else
        {
            result.push_back(3);
        }
    }
    std::cout << "classify success" << std::endl;
    std::cout << "result total: " << result.size() << std::endl;
    //convert result to result
    cv::Mat cls = convertToNewImage(result, image.rows, image.cols);
    cv::imwrite("fisher-plus.jpg", cls);
}