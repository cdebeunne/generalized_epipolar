#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Householder"
#include "eigen3/Eigen/QR"
#include "eigen3/Eigen/SVD"

#include "Camera.hpp"
#include "Timer.hpp"
#include "Epipolar.hpp"

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <ctime>



int main(int argc, char** argv){

    Timer timer;

    // initialize K
    Eigen::Matrix3f K;
    K << 458.654, 0, 367.215,
         0, 457.296, 248.375,
         0, 0, 1;
    double focal_length = K(0);
    cv::Point2f principal_pt(K(0,2), K(1,2));
    Camera cam(K);

    std::string image_path0 = "/home/cesar/Documents/phd/datasets/EUROC/MH_01_easy/mav0/cam0/data/1403636579763555584.png";
    std::string image_path1 = "/home/cesar/Documents/phd/datasets/EUROC/MH_01_easy/mav0/cam0/data/1403636579963555584.png";

    cv::Mat img_1 = cv::imread(image_path0, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(image_path1, cv::IMREAD_COLOR);

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    detector->detectAndCompute( img_1, cv::Mat(), keypoints_1, descriptors_1 );
    detector->detectAndCompute( img_2, cv::Mat(), keypoints_2, descriptors_2 );

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( descriptors_1, descriptors_2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //-- Draw only "good" matches
    cv::Mat img_matches;
    std::vector<cv::Point2d> kp_1_matched;
    std::vector<cv::Point2d> kp_2_matched;
    cv::drawMatches( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    imshow( "Good Matches", img_matches );
    for( int i = 0; i < (int)good_matches.size(); i++ )
    {
        printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
        kp_1_matched.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        kp_2_matched.push_back(keypoints_2[good_matches[i].queryIdx].pt);
    }
    cv::waitKey(0);

    // Threshold for RANSAC
    double thresh = std::atof(argv[1]);
    Eigen::Matrix3f best_E;
    std::cout << "threshold :" << thresh << std::endl;

    timer.start();
    EssentialRANSAC(kp_1_matched, kp_2_matched, cam, best_E, thresh);
    timer.stop();

    std::cout << "Elapsed time" << std::endl;
    std::cout << timer.elapsedSeconds() << std::endl; 
    std::cout << "Best Essential matrix" << std::endl;
    std::cout << best_E << std::endl;

    // recover displacement from E
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    recoverPose(best_E, cam, kp_1_matched, kp_2_matched, t, R);
    std::cout << "Translation" << std::endl;
    std::cout << t << std::endl;

    // Compare with opencv
    cv::Mat cvMask;
    timer.start();
    cv::Mat E_cv = cv::findEssentialMat(kp_1_matched, kp_2_matched, focal_length, principal_pt, cv::RANSAC, 0.95, 1.0, cvMask);
    timer.stop();
    std::cout << "Elapsed time" << std::endl;
    std::cout << timer.elapsedSeconds() << std::endl; 

    cv::Mat R_cv, t_cv;
    cv::Mat K_cv = (cv::Mat_<float>(3,3) << K(0,0), 0, K(0,2),
               0, K(1,1), K(1,2),
               0, 0, 1);  
    cv::recoverPose(E_cv, kp_1_matched, kp_2_matched, K_cv, R_cv, t_cv);
    std::cout << "Open CV essential Matrix" << std::endl;
    std::cout << E_cv << std::endl;
    std::cout << "Associated translation" << std::endl;
    std::cout << t_cv << std::endl;

    return 0;

}