#ifndef ASENSOR_HPP
#define ASENSOR_HPP

#include <opencv2/core.hpp>
#include "eigen3/Eigen/Core"

class ASensor
{
public:
    ASensor(Eigen::Matrix3d K): _K(K){};

    Eigen::Vector3d getRay(Eigen::Vector2d f) {
        // Get ray with the convention z front 
        
        Eigen::Vector3d rayCam; 
        rayCam[0] = (f(0) - _K(0,2)) / _K(0, 0);
        rayCam[1] = (f(1) - _K(1,2)) / _K(1, 1);
        rayCam[2] = 1;
        rayCam.normalize();
        return rayCam;
    }

    bool project(const Eigen::Vector3d &p3d, Eigen::Vector2d &p2d) {

        // to image homogeneous coordinates
        Eigen::Vector3d pt = _K * p3d;
        pt /= pt(2, 0);
        p2d = pt.block<2, 1>(0, 0);

        if (p3d[2] < 0.01) //point behind the camera
            return false;

        return true;
    }


protected:
    Eigen::Matrix3d _K;
};

#endif