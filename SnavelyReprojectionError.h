#ifndef SNAVELYREPROJECTIONERROR_H
#define SNAVELYREPROJECTIONERROR_H

#include <iostream>
#include <ceres/ceres.h>
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),
                                                                           observed_y(observation_y) {}

    template<typename T>
    bool operator() (const T *const camera,
                     const T *const point,
                     T *residuals) const {
        // camera[0, 1, 2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    /**
     * camera: 9 dims array
     * [0-2]: angle-axis rotation
     * [3-5]: translation
     * [6-8]: camera parameter,
     *        - [6] focal length,
     *        - [7-8] second and forth order radial distortion
     * point: 3D location
     * predictions: 2D predictions with center of the image plane
     */
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera,
                                                   const T *point,
                                                   T *predictions) {
        // Rodrigues' formula
        T p[3]; // T Matrix
        AngleAxisRotatePoint(camera, point, p);
        // camera[3, 4, 5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // compute the center of distortion
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and forth order radial distortion
        const T &l1 = camera[7]; // k1
        const T &l2 = camera[8]; // k2

        T r2 = xp * xp + yp * yp;
        // d = 1 + k_1 * r^2 + k_2 * r^4
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);
        // x_distorted = x_p * d
        // y_distorted = y_p * d

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