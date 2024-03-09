#include <iostream>
#include <Eigen/Dense>
#include<fstream>
#include <Eigen/LU>
#include <vector>

using namespace Eigen;
void load_data(std::ifstream &file, std::vector<std::vector<double>> &data)
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
int main() {
    std::string path = "data.txt"; // Replace with your file path
    std::ifstream file(path);
    std::vector<std::vector<double>> data;
    load_data(file, data);
    int numSamples = data.size();
    int numFeatures = data[0].size();

    int numRows1 = 52;
    int numRows2 = 48;

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

    std::cout << "总体类内离散度矩阵为：" << std::endl;
    std::cout << "sw12:\n" << sw12 << std::endl;
    std::cout << "sw13:\n" << sw13 << std::endl;
    std::cout << "sw23:\n" << sw23 << std::endl;

    // Discriminant functions and thresholds
    double T12 = -0.5 * (m1 + m2).dot(sw12.inverse() * a);
    double T13 = -0.5 * (m1 + m3).dot(sw13.inverse() * b);
    double T23 = -0.5 * (m2 + m3).dot(sw23.inverse() * c);

    int kind1 = 0, kind2 = 1, kind3 = 3;
    std::string test_path = "test.txt"; // Replace with your file path
    std::ifstream test_file(test_path);
    std::vector<std::vector<double>> test_data;
    load_data(test_file, test_data);
    int samples = test_data.size();
    int features = test_data[0].size();
    MatrixXd test(samples,features);
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < features; j++) {
            test(i, j) = test_data[i][j];
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
            result.push_back(kind1);
        }
        else if (g12 < 0 && g23 > 0) {
            result.push_back(kind1);
        }
        else if (g13 < 0 && g23 < 0) {
            result.push_back(kind1);
        }
    }
    
}