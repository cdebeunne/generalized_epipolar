#ifndef EPIPOLAR_HPP
#define EPIPOLAR_HPP

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Householder"
#include "eigen3/Eigen/QR"
#include "eigen3/Eigen/SVD"

#include "Camera.hpp"

#include <algorithm>
#include <vector>
#include <numeric>
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

Eigen::Vector3f triangulate(Eigen::Vector3f ray0, Eigen::Vector3f ray1, Eigen::Vector3f t){
    // triangulate point with mid point method

    // Get ray and optical centers of cameras in world coordinates
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
    Eigen::Vector3f C(0,0,0);

    // Process the rays
    Eigen::Matrix3f A;
    Eigen::Vector3f o;

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

    // Process landmark pose in camera frame x in front !
    Eigen::Vector3f position = S.inverse()*C;
    return position;
}

void recoverPose(Eigen::Matrix3f E, Camera &cam, std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, Eigen::Vector3f &t, Eigen::Matrix3f &R){
    // recover displacement from E
    // We then have x2 = Rx1 + t
    Eigen::JacobiSVD<Eigen::MatrixXf> svd_2(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f Rzp;
    Rzp << 0, 1, 0,
        -1, 0, 0,
        0, 0, 1;
    
    R = svd_2.matrixU() * Rzp.transpose() * svd_2.matrixV();
    Eigen::Matrix3f tx = svd_2.matrixU() * Rzp * svd_2.singularValues().asDiagonal() * svd_2.matrixU();
    t << tx(2,1), tx(0,2), tx(1,0);

    // Let's see if we have positive or negative depth
    float avg_depth = 0;
    for (int i=0; i < (int)kp_1_matched.size(); i++){
        Eigen::Vector3f lmk;
        float u0 = kp_1_matched[i].x;
        float v0 = kp_1_matched[i].y;
        float u1 = kp_2_matched[i].x;
        float v1 = kp_2_matched[i].y;
        lmk = triangulate(cam.getRay(u0, v0), cam.getRay(u1, v1), t);
        avg_depth = avg_depth + lmk(2);
    }
    if (avg_depth < 0){
        Eigen::Matrix3f Rzm;
        Rzm << 0, -1, 0,
            1, 0, 0,
            0, 0, 1;
        
        R = svd_2.matrixU() * Rzm.transpose() * svd_2.matrixV();
        tx = svd_2.matrixU() * Rzm * svd_2.singularValues().asDiagonal() * svd_2.matrixU();
        t << tx(2,1), tx(0,2), tx(1,0);

        // Let's see if we have positive or negative depth
        for (int i=0; i < (int)kp_1_matched.size(); i++){
            Eigen::Vector3f lmk;
            float u0 = kp_1_matched[i].x;
            float v0 = kp_1_matched[i].y;
            float u1 = kp_2_matched[i].x;
            float v1 = kp_2_matched[i].y;
            lmk = triangulate(cam.getRay(u0, v0), cam.getRay(u1, v1), t);
        }
    }
}

void EssentialRANSAC(std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, Camera &cam, Eigen::Matrix3f &best_E, float threshold){
    int best_number_of_inliers = 0;
    float w = 0.5;
    float T = std::log(1-0.999)/std::log(1-std::pow(w, 8));

    for( int k=0; k<T; k++){

        std::vector<int> index_list = random_index((int)kp_1_matched.size()); 

        // Let's find the essential matrix with the 8 points algorithm
        
        Eigen::MatrixXf A = Eigen::MatrixXf::Zero(9,8);
        Eigen::VectorXf b = Eigen::VectorXf::Zero(9);

        for(int i=0; i<8; i++){
            cv::Point2f x1 = kp_1_matched[index_list[i]];
            cv::Point2f x2 = kp_2_matched[index_list[i]];
            
            Eigen::Vector3f x1v = cam.getRay(x1.x, x2.y);
            Eigen::Vector3f x2v = cam.getRay(x2.x, x2.y);

            A.col(i) << x1v.x()*x2v.x(), x2v.x()*x1v.y(), x2v.x()*1,
                        x2v.y()*x1v.x(), x2v.y()*x1v.y(), x2v.y()*1,
                        1*x1v.x(), 1*x1v.y(), 1*1;
        }

        // Step 1: compute a first approximation of E

        // Compute the eigen values of A * A^T
        Eigen::JacobiSVD<Eigen::MatrixXf> svd_0(A*A.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Compute the approximated Essential matrix
        Eigen::VectorXf e = svd_0.matrixU().col(8);
        Eigen::Matrix3f E;
        E << e(0), e(1), e(2),
            e(3), e(4), e(5),
            e(6), e(7), e(8);

        // Step 2: project it into the essential space
        Eigen::JacobiSVD<Eigen::MatrixXf> svd_1(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3f SIGMA;
        SIGMA << 1, 0, 0,
                0, 1, 0,
                0, 0, 0;

        E = svd_1.matrixU() * SIGMA * svd_1.matrixV();

        // Step 3: Check inliers
        float score;
        int nb_inliers = 0;
        for (int i = 0; i < (int)kp_1_matched.size(); i++ ){
            cv::Point2f x1 = kp_1_matched[index_list[i]];
            cv::Point2f x2 = kp_2_matched[index_list[i]];

            Eigen::Vector3f x1v = cam.getRay(x1.x, x2.y);
            Eigen::Vector3f x2v = cam.getRay(x2.x, x2.y);

            score = std::abs(x2v.transpose() * E * x1v);
            if (score < threshold) nb_inliers++;
        }
        
        // Step 4: Update the Essential Matrix
        if (nb_inliers > best_number_of_inliers){
            best_number_of_inliers = nb_inliers;
            best_E = E;
        }

        // Step 5: Recompute T
        float w_est = (float)best_number_of_inliers / (float)kp_1_matched.size();
        if (w_est > w){
            T = std::log(1-0.999)/std::log(1-std::pow(w_est, 8));
            w = w_est;
        }
    }
}

#endif