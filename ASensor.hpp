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

        rayCam.segment<2>(0) = (f - _K.block<2, 1>(0, 2));
        rayCam[0] /= _K(0, 0);
        rayCam[1] /= _K(1, 1);
        rayCam[2] = 1;
        rayCam.normalize();
        return rayCam;
    }


protected:
    Eigen::Matrix3d _K;
};

#endif