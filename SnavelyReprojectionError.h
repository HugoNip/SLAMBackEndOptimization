#ifndef SNAVELYREPROJECTIONERROR_H
#define SNAVELYREPROJECTIONERROR_H

// used in ceres-BA method

#include <iostream>
#include <ceres/ceres.h>
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y) :
        observed_x(observation_x), observed_y(observation_y) {}

    template<typename T>
    bool operator() (const T *const camera,
                     const T *const point,
                     T *residuals) const {

        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        // (E9.41) e = z - h(T, p)
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    /**
     * Projection
     * p -> z
     * p = [X, Y, Z] is 3D points in global coordinate
     * z = [u, v] is pixel coordinate
     *
     * camera: 9 dims array
     * [0-2]: angle-axis for rotation
     * [3-5]: translation
     * [6-8]: camera parameter,
     *        - [6] focal length,
     *        - [7-8] second and forth order radial distortion
     * point: 3D location
     * predictions: 2D predictions for the center of the image plane
     */
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, // 9D
                                                   const T *point,
                                                   T *predictions) {
        // Step 1: Compute the projection in Camera coordinate
        // P_c
        // Rodrigues' formula
        T p[3]; // T Matrix, converted from points
        AngleAxisRotatePoint(camera, point, p); // Rp

        // P' = Rp + t
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // compute the center of distortion
        // P_c = [P'_X/P'_Z, P'_Y/P'_Z, 1]
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Step 2: Compute the original coordinate in the pre-undistort
        // [u'_c, v'_c]
        // Apply second and forth order radial distortion
        const T &l1 = camera[7]; // k1
        const T &l2 = camera[8]; // k2

        T r2 = xp * xp + yp * yp;
        // d = 1 + k_1 * r^2 + k_2 * r^4
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);
        // x_distorted = x_p * d
        // y_distorted = y_p * d

        // Step 3: Compute pixel coordinate
        // [u_s, v_s]
        const T &focal = camera[6]; // f_x, f_y
        // u = f_x * x_distorted
        // v = f_y * y_distorted
        predictions[0] = focal * distortion * xp; // u
        predictions[1] = focal * distortion * yp; // v

        return true;
    }

    // use auto diff cost function
    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

#endif // SNAVELYREPROJECTIONERROR_H