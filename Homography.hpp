#ifndef HOMOGRAPHY_HPP
#define HOMOGRAPHY_HPP

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Householder"
#include "eigen3/Eigen/QR"
#include "eigen3/Eigen/SVD"

#include "ASensor.hpp"

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <memory>
#include <random>


std::vector<int> random_index(int size){
    std::vector<int> index_list;
    index_list.resize(size, 0);
    std::iota(index_list.begin(), index_list.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
 
    std::shuffle(index_list.begin(), index_list.end(), g);
    return index_list;
}

void HomographyRANSAC(std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, std::shared_ptr<ASensor> &cam, Eigen::Matrix3d &best_H, float threshold, std::vector<int> &inliers){
    double best_score = 0;
    // float w = 0.5;
    // float T = std::log(1-0.999)/std::log(1-std::pow(w, 8));
    std::vector<int> inliers_iter; // 1 if in, 0 if out  

    for( int k=0; k<4000; k++){
        std::vector<int> index_list = random_index((int)kp_1_matched.size());

        // Let's find the Homography using 8 points
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2*8,9);

        for(int i=0; i<8; i++){
            cv::Point2d x1 = kp_1_matched[index_list[i]];
            cv::Point2d x2 = kp_2_matched[index_list[i]];
            
            Eigen::Vector3d x1v = cam->getRay(Eigen::Vector2d(x1.x, x1.y));
            Eigen::Vector3d x2v = cam->getRay(Eigen::Vector2d(x2.x, x2.y));

            A(2*i,0) = 0.0;
            A(2*i,1) = 0.0;
            A(2*i,2) = 0.0;
            A(2*i,3) = -x1v(2) * x2v(0);
            A(2*i,4) = -x1v(2) * x2v(1);
            A(2*i,5) = -x1v(2) * x2v(2);
            A(2*i,6) = x1v(1) * x2v(0);
            A(2*i,7) = x1v(1) * x2v(1);
            A(2*i,8) = x1v(1) * x2v(2);
            A(2*i+1,0) = x1v(2) * x2v(0);
            A(2*i+1,1) = x1v(2) * x2v(1);
            A(2*i+1,2) = x1v(2) * x2v(2);
            A(2*i+1,3) = 0.0;
            A(2*i+1,4) = 0.0;
            A(2*i+1,5) = 0.0;
            A(2*i+1,6) = -x1v(0) * x2v(0);
            A(2*i+1,7) = -x1v(0) * x2v(1);
            A(2*i+1,8) = -x1v(0) * x2v(2);
        }

        // Compute the eigen values of A
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_0(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Compute the approximated Essential matrix
        Eigen::VectorXd h = svd_0.matrixV().col(8);
        Eigen::Matrix3d H12;
        H12 << h(0), h(1), h(2),
            h(3), h(4), h(5),
            h(6), h(7), h(8);
        
        // Check inliers
        double score = 0;
        inliers_iter.clear(); 
        for (int i = 0; i < (int)kp_1_matched.size(); i++ ){
            cv::Point2d x1 = kp_1_matched[i];
            cv::Point2d x2 = kp_2_matched[i];

            Eigen::Vector3d x1v = cam->getRay(Eigen::Vector2d(x1.x, x1.y));
            Eigen::Vector3d x2v = cam->getRay(Eigen::Vector2d(x2.x, x2.y));

            // residuals are 1 - cos of the angle between the predicted ray and the ray
            Eigen::Vector3d x2v_in1 = H12 * x2v;
            double residual_1 = 1.0 - x2v_in1.dot(x1v) / x2v_in1.norm();  

            if (threshold < residual_1){
                inliers_iter.push_back(0);
                continue;
            } else score += residual_1 + 1;

            Eigen::Vector3d x1v_in2 = H12.inverse() * x1v;
            double residual_2 = 1.0 - x1v_in2.dot(x2v) / x1v_in2.norm();

            if (threshold < residual_2){
                inliers_iter.push_back(0);
                continue;
            } else {
                inliers_iter.push_back(1);
                score += residual_2 + 1;
            }
        }

        // Update Matrix
        if(score > best_score){
            best_score = score;
            best_H = H12;
            inliers = inliers_iter;
        }
    }
}


#endif