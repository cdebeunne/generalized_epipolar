#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/core.hpp>
#include "eigen3/Eigen/Core"

class Camera
{
public:
    Camera(Eigen::Matrix3f K): _K(K){};

    Eigen::Vector3f getRay(Eigen::Vector2f f) {
        // Get ray with the convention z front 
        
        Eigen::Vector3f rayCam;

        rayCam.segment<2>(0) = (f - _K.block<2, 1>(0, 2));
        rayCam[0] /= _K(0, 0);
        rayCam[1] /= _K(1, 1);
        rayCam[2] = 1;
        return rayCam;
    }


protected:
    Eigen::Matrix3f _K;
};

#endif