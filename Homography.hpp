#ifndef HOMOGRAPHY_HPP
#define HOMOGRAPHY_HPP

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Householder"
#include "eigen3/Eigen/QR"
#include "eigen3/Eigen/SVD"

#include <opencv2/core.hpp>

#include "ASensor.hpp"
#include "Epipolar.hpp"

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <memory>
#include <random>

Eigen::Vector3f rotationMatrixToEulerAngles(cv::Mat &R)
{

    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Eigen::Vector3f(x*180/3.1416, y*180/3.1416, z*180/3.1416);

}


void HomographyRANSAC(std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, std::shared_ptr<ASensor> &cam, Eigen::Matrix3d &best_H, float threshold, 
                        std::vector<int> &inliers, int NPoints = 8, int Niter = 4000){
    double best_score = 0;
    std::vector<int> inliers_iter; // 1 if in, 0 if out  

    for( int k=0; k<Niter; k++){
        std::vector<int> index_list = random_index((int)kp_1_matched.size());

        // Let's find the Homography using 8 points
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2*NPoints,9);

        for(int i=0; i<NPoints; i++){
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
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_0(A, Eigen::ComputeFullV);

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

bool recoverPoseHomography(std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, std::shared_ptr<ASensor> &cam, Eigen::Matrix3d &H, Eigen::Vector3d &t, Eigen::Matrix3d &R)
{
    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d Vt = V.transpose();
    Eigen::Vector3d w = svd.singularValues();

    double s = U.determinant() * Vt.determinant();

    double d1 = w(0);
    double d2 = w(1);
    double d3 = w(2);
    
    // We ignore solutions with multiplicity
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    std::vector<Eigen::Matrix3d> vR;
    std::vector<Eigen::Vector3d> vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    double aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    double aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    double x1[] = {aux1,aux1,-aux1,-aux1};
    double x3[] = {aux3,-aux3,aux3,-aux3};

    //case d' > 0 ie d' = d2
    double aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    double ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    double stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        Eigen::Matrix3d Rp;
        Rp.setZero();
        Rp(0,0) = ctheta;
        Rp(0,2) = -stheta[i];
        Rp(1,1) = 1.0;
        Rp(2,0) = stheta[i];
        Rp(2,2) = ctheta;

        Eigen::Matrix3d R = s*U*Rp*Vt;
        vR.push_back(R);

        Eigen::Vector3d tp;
        tp(0) = x1[i];
        tp(1) = 0;
        tp(2) = -x3[i];
        tp *= d1-d3;

        Eigen::Vector3d t = U*tp;
        vt.push_back(t / t.norm());

        Eigen::Vector3d np;
        np(0) = x1[i];
        np(1) = 0;
        np(2) = x3[i];

        Eigen::Vector3d n = V*np;
        if(n(2) < 0)
            n = -n;
        vn.push_back(n);
    }

    //case d'=-d2
    double aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    double cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    double sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        Eigen::Matrix3d Rp;
        Rp.setZero();
        Rp(0,0) = cphi;
        Rp(0,2) = sphi[i];
        Rp(1,1) = -1;
        Rp(2,0) = sphi[i];
        Rp(2,2) = -cphi;

        Eigen::Matrix3d R = s*U*Rp*Vt;
        vR.push_back(R);

        Eigen::Vector3d tp;
        tp(0) = x1[i];
        tp(1) = 0;
        tp(2) = x3[i];
        tp *= d1+d3;

        Eigen::Vector3d t = U*tp;
        vt.push_back(t / t.norm());

        Eigen::Vector3d np;
        np(0) = x1[i];
        np(1) = 0;
        np(2) = x3[i];

        Eigen::Vector3d n = V*np;
        if(n(2) < 0)
            n = -n;
        vn.push_back(n);
    }
    int bestInliers = 0;
    int secondBestInliers = 0;
    int bestId = 0;

    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        std::vector<int> inliers;
        int nInliers = checkRT(cam, vR[i], vt[i], kp_1_matched, kp_2_matched, inliers);
        std::cout << "NInliers for R :" << std::endl;
        std::cout << vR[i] << std::endl;
        std::cout << " And t :" << std::endl;
        std::cout << vt[i] << std::endl;
        std::cout << nInliers << std::endl;

        if(nInliers>bestInliers)
        {
            secondBestInliers = bestInliers;
            bestInliers = nInliers;
            bestId = i;
        }
        else if(nInliers>secondBestInliers)
        {
            secondBestInliers = nInliers;
        }
    }


    if(secondBestInliers<=bestInliers)
    {
        R = vR[bestId];
        t = vt[bestId];
        return true;
    }

    return false;
}

bool estimateMotionWithHomographyCV(std::vector<cv::Point2d> kp_1_matched, std::vector<cv::Point2d> kp_2_matched, cv::Mat K, Eigen::Vector3d &t, Eigen::Matrix3d &R)
{
    cv::Mat cvMask, H;
    H = cv::findHomography(kp_1_matched, kp_2_matched, cv::RANSAC, 3, cvMask);
    H /= H.at<double>(2, 2);

    // Get inliers
    std::vector<int> inliers;
    for (int i = 0; i < cvMask.rows; i++)
        if ((int)cvMask.at<unsigned char>(i, 0) == 1)
            inliers.push_back(i);

    // -- Recover R,t from Homograph matrix
    std::vector<cv::Mat> Rs, ts, normals;
    cv::decomposeHomographyMat(H, K, Rs, ts, normals);
    // Normalize t
    for (auto &t : ts)
    {
        t = t / sqrt(t.at<double>(1, 0) * t.at<double>(1, 0) + t.at<double>(2, 0) * t.at<double>(2, 0) +
                        t.at<double>(0, 0) * t.at<double>(0, 0));
    }

    // Remove wrong RT
    // If for a (R,t), a point's pos is behind the camera, then this is wrong.
    std::vector<cv::Mat> res_Rs, res_ts, res_normals;
    cv::Mat possibleSolutions; //Use print_MatProperty to know its type: 32SC1
    std::vector<cv::Point2f> kp_1_matched_np, kp_2_matched_np;
    for (int idx : inliers)
    {
        kp_1_matched_np.push_back(cv::Point2f((kp_1_matched[idx].x-K.at<double>(0,2))/K.at<double>(0,0), (kp_1_matched[idx].y-K.at<double>(1,2))/K.at<double>(1,1)));
        kp_2_matched_np.push_back(cv::Point2f((kp_2_matched[idx].x-K.at<double>(0,2))/K.at<double>(0,0), (kp_2_matched[idx].y-K.at<double>(1,2))/K.at<double>(1,1)));
    }

    cv::filterHomographyDecompByVisibleRefpoints(Rs, normals, kp_1_matched_np, kp_2_matched_np, possibleSolutions);
    for (int i = 0; i < possibleSolutions.rows; i++)
    {
        std::cout << "HOMOCV " << i << std::endl;
        int idx = possibleSolutions.at<int>(i, 0);
        res_Rs.push_back(Rs[idx]);
        res_ts.push_back(ts[idx]);
        res_normals.push_back(normals[idx]);
        std::cout << Rs[idx] << std::endl;
        std::cout << ts[idx] << std::endl;
    }
    return true;

}




#endif