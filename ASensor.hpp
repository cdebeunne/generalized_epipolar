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

        if (p3d[2] < 0.01) //point behind the camera
            return false;
        
        p2d(0) = _K(0,0)*p3d(0)/p3d(2) + _K(0,2);
        p2d(1) = _K(1,1)*p3d(1)/p3d(2) + _K(1,2);

        return true;
    }


protected:
    Eigen::Matrix3d _K;
};

#endif