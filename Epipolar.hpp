#ifndef EPIPOLAR_HPP
#define EPIPOLAR_HPP

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

Eigen::Vector3d triangulate(Eigen::Vector3d ray0, Eigen::Vector3d ray1, Eigen::Vector3d t){
    // triangulate point with mid point method

    // Get ray and optical centers of cameras
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    Eigen::Vector3d C(0,0,0);

    // Process the rays
    Eigen::Matrix3d A;
    Eigen::Vector3d o;

    // ray cam 0
    A <<
        ray0[0]*ray0[0] - 1,ray0[0]*ray0[1],ray0[0]*ray0[2],
        ray0[0]*ray0[1],ray0[1]*ray0[1] - 1,ray0[1]*ray0[2],
        ray0[0]*ray0[2],ray0[1]*ray0[2],ray0[2]*ray0[2] - 1;
    o << 0, 0, 0;
    S += A;
    C += A*o;

    // ray cam 1 
    A <<
        ray1[0]*ray1[0] - 1,ray1[0]*ray1[1],ray1[0]*ray1[2],
        ray1[0]*ray1[1],ray1[1]*ray1[1] - 1,ray1[1]*ray1[2],
        ray1[0]*ray1[2],ray1[1]*ray1[2],ray1[2]*ray1[2] - 1;
    o = t;
    S += A;
    C += A*o;

    // Process landmark pose in camera frame 
    Eigen::Vector3d position = S.inverse()*C;
    return position;
}

void recoverPose(Eigen::Matrix3d E, std::shared_ptr<ASensor> &cam, std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, Eigen::Vector3d &t, Eigen::Matrix3d &R, std::vector<int> inliers){
    // recover displacement from E
    // We then have x2 = Rx1 + t
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_2(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d Rzp;
    Rzp << 0, -1, 0,
        1, 0, 0,
        0, 0, 1;
    std::vector<int> new_inliers;
    
    R = svd_2.matrixU() * Rzp.transpose() * svd_2.matrixV().transpose();
    Eigen::Matrix3d tx = svd_2.matrixU() * Rzp * svd_2.singularValues().asDiagonal() * svd_2.matrixU().transpose();
    t << tx(2,1), tx(0,2), tx(1,0);

    // Let's see if we have positive or negative depth
    float avg_depth = 0;
    for (int i=0; i < (int)kp_1_matched.size(); i++){
        if (inliers[i] == 0){
            continue;
        }
        float u0 = kp_1_matched[i].x;
        float v0 = kp_1_matched[i].y;
        float u1 = kp_2_matched[i].x;
        float v1 = kp_2_matched[i].y;
        Eigen::Vector3d lmk = triangulate(cam->getRay(Eigen::Vector2d(u0, v0)), cam->getRay(Eigen::Vector2d(u1, v1)), t);
        avg_depth = avg_depth + lmk(2);

        // check inlier
        if(lmk(2) < 0){
            new_inliers.push_back(0);
        } else{
            new_inliers.push_back(1);
        }
    }
    if (avg_depth < 0){
        Eigen::Matrix3d Rzm;
        Rzm << 0, 1, 0,
            -1, 0, 0,
            0, 0, 1;
        
        R = svd_2.matrixU() * Rzm.transpose() * svd_2.matrixV().transpose();
        tx = svd_2.matrixU() * Rzm * svd_2.singularValues().asDiagonal() * svd_2.matrixU().transpose();
        t << tx(2,1), tx(0,2), tx(1,0);

        // Let's see if we have positive or negative depth
        new_inliers.clear();
        for (int i=0; i < (int)kp_1_matched.size(); i++){
            if (inliers[i] == 0){
                continue;
            }
            float u0 = kp_1_matched[i].x;
            float v0 = kp_1_matched[i].y;
            float u1 = kp_2_matched[i].x;
            float v1 = kp_2_matched[i].y;
            Eigen::Vector3d lmk = triangulate(cam->getRay(Eigen::Vector2d(u0, v0)), cam->getRay(Eigen::Vector2d(u1, v1)), t);

            // check inlier
            if(lmk(2) < 0){
                new_inliers.push_back(0);
            } else{
                new_inliers.push_back(1);
            }
        }
    }
    inliers = new_inliers;
}

void EssentialRANSAC(std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, std::shared_ptr<ASensor> &cam, Eigen::Matrix3d &best_E, float threshold, std::vector<int> &inliers){
    int best_number_of_inliers = 0;
    float w = 0.5;
    float T = std::log(1-0.9999)/std::log(1-std::pow(w, 8));
    std::vector<int> inliers_iter; // 1 if in, 0 if out  

    for( int k=0; k<T; k++){

        std::vector<int> index_list = random_index((int)kp_1_matched.size());
        inliers_iter.clear(); 

        // Let's find the essential matrix with the 8 points algorithm
        
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(8,9);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(9);

        for(int i=0; i<8; i++){
            cv::Point2d x1 = kp_1_matched[index_list[i]];
            cv::Point2d x2 = kp_2_matched[index_list[i]];
            
            Eigen::Vector3d x1v = cam->getRay(Eigen::Vector2d(x1.x, x1.y));
            Eigen::Vector3d x2v = cam->getRay(Eigen::Vector2d(x2.x, x2.y));

            A(i,0) = x1v.x()*x2v.x();
            A(i,1) = x2v.x()*x1v.y();
            A(i,2) = x2v.x()*1;
            A(i,3) = x2v.y()*x1v.x();
            A(i,4) = x2v.y()*x1v.y();
            A(i,5) = x2v.y()*1;
            A(i,6) = 1*x1v.x();
            A(i,7) = 1*x1v.y();
            A(i,8) = 1*1;
        }

        // Step 1: compute a first approximation of E

        // Compute the eigen values of A
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_0(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Compute the approximated Essential matrix
        Eigen::VectorXd e = svd_0.matrixV().col(8);
        Eigen::Matrix3d E;
        E << e(0), e(1), e(2),
            e(3), e(4), e(5),
            e(6), e(7), e(8);

        // Step 2: project it into the essential space
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_1(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d SIGMA;
        SIGMA << 1, 0, 0,
                0, 1, 0,
                0, 0, 0;

        E = svd_1.matrixU() * SIGMA * svd_1.matrixV().transpose();

        // Step 3: Check inliers
        float score;
        int nb_inliers = 0;
        for (int i = 0; i < (int)kp_1_matched.size(); i++ ){
            cv::Point2d x1 = kp_1_matched[i];
            cv::Point2d x2 = kp_2_matched[i];

            Eigen::Vector3d x1v = cam->getRay(Eigen::Vector2d(x1.x, x1.y));
            Eigen::Vector3d x2v = cam->getRay(Eigen::Vector2d(x2.x, x2.y));

            // Score computed with Sampson's error
            score = std::abs(x2v.transpose() * E * x1v) / std::sqrt(
                (x2v.transpose() * E)(0) * (x2v.transpose() * E)(0) +
                (x2v.transpose() * E)(1) * (x2v.transpose() * E)(1) +
                (E * x1v)(0) * (E * x1v)(0) +
                (E * x1v)(1) * (E * x1v)(1));

            if (score < threshold){
                nb_inliers++;
                inliers_iter.push_back(1);
            } else{
                inliers_iter.push_back(0);
            }
        }
        
        // Step 4: Update the Essential Matrix
        if (nb_inliers > best_number_of_inliers){
            best_number_of_inliers = nb_inliers;
            best_E = E;
            inliers = inliers_iter;
        }

        // Step 5: Recompute T
        float w_est = (float)best_number_of_inliers / (float)kp_1_matched.size();
        if (w_est > w){
            T = std::log(1-0.9999)/std::log(1-std::pow(w_est, 8));
            w = w_est;
        }
    }
}

#endif