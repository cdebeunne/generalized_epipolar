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

Eigen::Vector3d rotationMatrixToEulerAnglesEigen(Eigen::Matrix3d &R)
{

    float sy = sqrt(R(0,0) * R(0,0) +  R(1,0) * R(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R(2,1) , R(2,2));
        y = atan2(-R(2,0), sy);
        z = atan2(R(1,0), R(0,0));
    }
    else
    {
        x = atan2(-R(1,2), R(1,1));
        y = atan2(-R(2,0), sy);
        z = 0;
    }
    return Eigen::Vector3d(x*180/3.1416, y*180/3.1416, z*180/3.1416);

}


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

int checkRT(std::shared_ptr<ASensor> &cam, const Eigen::Matrix3d &R, const Eigen::Vector3d &t, std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, std::vector<int> inliers){

    int inliers_number = 0;
    inliers.clear();

    // We check if the depth is positive
    for (int i=0; i < (int)kp_1_matched.size(); i++){
        float u0 = kp_1_matched[i].x;
        float v0 = kp_1_matched[i].y;
        float u1 = kp_2_matched[i].x;
        float v1 = kp_2_matched[i].y;
        Eigen::Vector3d normal1 = cam->getRay(Eigen::Vector2d(u0, v0));
        Eigen::Vector3d normal2 = cam->getRay(Eigen::Vector2d(u1, v1));
        Eigen::Vector3d lmk_C1 = triangulate(normal1, normal2, t);

        // check parallax
        double dist1 = normal1.norm();
        double dist2 = normal2.norm();

        double cosParallax = normal1.dot(normal2) / (dist1 * dist2);

        if(!std::isfinite(lmk_C1(0)) || !std::isfinite(lmk_C1(1)) || !std::isfinite(lmk_C1(2)))
        {
            inliers.push_back(0);
            continue;
        }

        // check depth wrt C1 only if enough parallax as infinite point can have negative depth
        if(lmk_C1(2) <= 0 || cosParallax > 0.99998){
            inliers.push_back(0);
            continue;
        } 

        // check depth wrt C2 as well
        Eigen::Vector3d lmk_C2 = R * lmk_C1 + t;
        if(lmk_C2(2) <= 0 || cosParallax > 0.99998){
            inliers.push_back(0);
            continue;
        }
        inliers.push_back(1);
        inliers_number++;
    }

    return inliers_number;
}

bool recoverPoseEssential(Eigen::Matrix3d E, std::shared_ptr<ASensor> &cam, std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, Eigen::Vector3d &t, Eigen::Matrix3d &R, std::vector<int> inliers){
    // recover displacement from E
    // We then have x2 = Rx1 + t
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d Vt = svd.matrixV().transpose();

    // Let's compute the possible rotation and translation 
    Eigen::Matrix3d W;
    W << 0, 1, 0,
        -1, 0, 0,
        0, 0, 1;
    std::vector<int> new_inliers;
    
    Eigen::Matrix3d R1 = U * W * Vt;
    if (R1.determinant() < 0)
            R1 = -R1;
    Eigen::Vector3d t1 = U.col(2);
    t1 = t1 / t1.norm();

    Eigen::Matrix3d R2 = U * W.transpose() * Vt;
    if (R2.determinant() < 0)
            R2 = -R2;
    Eigen::Vector3d t2 = -t1;
    int nInliers1, nInliers2, nInliers3, nInliers4;
    std::vector<int> inliers1, inliers2, inliers3, inliers4;

    Eigen::Vector3d rpy_R1 = rotationMatrixToEulerAnglesEigen(R1);
    if (std::abs(rpy_R1(0)) < 90 && std::abs(rpy_R1(1)) < 90 && std::abs(rpy_R1(2)) < 90){
        nInliers1 = checkRT(cam, R1, t1, kp_1_matched, kp_2_matched, inliers1);
        nInliers2 = checkRT(cam, R1, t2, kp_1_matched, kp_2_matched, inliers2);
    } else{
        nInliers1 = 0;
        nInliers2 = 0;
    }

    Eigen::Vector3d rpy_R2 = rotationMatrixToEulerAnglesEigen(R2);
    if (std::abs(rpy_R2(0)) < 90 && std::abs(rpy_R2(1)) < 90 && std::abs(rpy_R2(2)) < 90){
        nInliers3 = checkRT(cam, R2, t1, kp_1_matched, kp_2_matched, inliers3);
        nInliers4 = checkRT(cam, R2, t2, kp_1_matched, kp_2_matched, inliers4);
    } else{
        nInliers3 = 0;
        nInliers4 = 0;
    }

    int maxInliers = std::max(nInliers1,std::max(nInliers2, std::max(nInliers3, nInliers4)));
    if (maxInliers == 0) return false;

    // Select the transformation with the biggest nInliers
    if (maxInliers == nInliers1){
        R = R1;
        t = t1;
        inliers = inliers1;
    } else if (maxInliers == nInliers2){
        R = R1;
        t = t2;
        inliers = inliers2;
    } else if (maxInliers == nInliers3){
        R = R2;
        t = t1;
        inliers = inliers3;
    } else if (maxInliers == nInliers4){
        R = R2;
        t = t2;
        inliers = inliers4;
    }
    return true;
}

void EssentialRANSAC(std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, std::shared_ptr<ASensor> &cam, Eigen::Matrix3d &best_E,
                     std::vector<int> &inliers, int Npoints=8, int Niter=4000){
    double best_score = 0;
    std::vector<int> inliers_iter; // 1 if in, 0 if out  

    for( int k=0; k<Niter; k++){

        std::vector<int> index_list = random_index((int)kp_1_matched.size());

        // Let's find the essential matrix with the 8 points algorithm
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Npoints,9);

        for(int i=0; i<Npoints; i++){
            cv::Point2d x1 = kp_1_matched[index_list[i]];
            cv::Point2d x2 = kp_2_matched[index_list[i]];
            
            Eigen::Vector3d x1v = cam->getRay(Eigen::Vector2d(x1.x, x1.y));
            Eigen::Vector3d x2v = cam->getRay(Eigen::Vector2d(x2.x, x2.y));

            A(i,0) = x2v.x()*x1v.x();
            A(i,1) = x2v.x()*x1v.y();
            A(i,2) = x2v.x()*x1v.z();
            A(i,3) = x2v.y()*x1v.x();
            A(i,4) = x2v.y()*x1v.y();
            A(i,5) = x2v.y()*x1v.z();
            A(i,6) = x2v.z()*x1v.x();
            A(i,7) = x2v.z()*x1v.y();
            A(i,8) = x2v.z()*x1v.z();
        }

        // Step 1: compute a first approximation of E

        // Compute the eigen values of A
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_0(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Compute the approximated Essential matrix
        Eigen::VectorXd e = svd_0.matrixV().col(8);
        Eigen::Matrix3d E_init;
        E_init << e(0), e(1), e(2),
            e(3), e(4), e(5),
            e(6), e(7), e(8);


        // Step 2: project it into the essential space
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_1(E_init, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d SIGMA;
        SIGMA << 1, 0, 0,
                0, 1, 0,
                0, 0, 0;

        Eigen::Matrix3d E = svd_1.matrixU() * SIGMA * svd_1.matrixV().transpose();

        // Step 3: Check inliers
        double score = 0;
        double threshold = 0.008;
        inliers_iter.clear(); 
        for (int i = 0; i < (int)kp_1_matched.size(); i++ ){
            cv::Point2d x1 = kp_1_matched[i];
            cv::Point2d x2 = kp_2_matched[i];

            Eigen::Vector3d x1v = cam->getRay(Eigen::Vector2d(x1.x, x1.y));
            Eigen::Vector3d x2v = cam->getRay(Eigen::Vector2d(x2.x, x2.y));

            // Residuals computed with the angle wrt to epiplanes
            Eigen::Vector3d epiplane_1 = E * x1v;
            double residual_1 = std::abs(epiplane_1.dot(x2v)) / epiplane_1.norm();

            if(threshold < residual_1){
                inliers_iter.push_back(0);
                continue;
            } else
                score += (threshold - residual_1) * (threshold - residual_1);

            Eigen::Vector3d epiplane_2 = E.transpose() * x2v;
            double residual_2 = std::abs(epiplane_2.dot(x1v)) / epiplane_2.norm();

            if(threshold < residual_2){
                inliers_iter.push_back(0);
                continue;
            } else {
                score += (threshold - residual_2) * (threshold - residual_2);
                inliers_iter.push_back(1);
            }
        }
        
        // Step 4: Update the Essential Matrix
        if (score> best_score){
            best_score = score;
            best_E = E;
            inliers = inliers_iter;
        }
    }
}

#endif
