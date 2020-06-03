#pragma once

#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common.h"
#include "rotation.h"
#include "random.h"

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

// read file from BAL dataset
class BALProblem {
public:
    // load bal data from text file
    explicit BALProblem(const std::string &filename, bool use_quaternions = false);

    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    // save results to text file
    void WriteToFile(const std::string &filename) const;

    // save results to ply pointcloud
    void WriteToPLYFile(const std::string &filename) const;

    void Normalize();

    void Perturb(const double rotation_sigma,
            const double translation_sigma,
            const double point_sigma);

    /**
     * 10: R = [q_0, q_1, q_2, q_3]
     * 9: R = [theta_1, theta_2, theta_3]
     */
    int camera_block_size() const {  return use_quaternions_ ? 10 : 9;  }

    int point_block_size() const {  return 3;  } // 3

    int num_cameras() const {  return num_cameras_;  }

    int num_points() const {  return num_points_;  }

    int num_observations() const {  return num_observations_;  } // z

    int num_parameters() const {  return num_parameters_;  }

    const int *point_index() const {  return point_index_;  }

    const int *camera_index() const {  return camera_index_;  }

    const double *observations() const {  return observations_;  } // z

    const double *parameters() const {  return parameters_;  }

    const double *cameras() const {  return parameters_;  }

    const double *points() const {  return parameters_ + camera_block_size() * num_cameras_;  }

    // camera parameters start address
    double *mutable_cameras() {  return parameters_;  }

    double *mutable_points() {  return parameters_ + camera_block_size() * num_cameras_;  }

    double *mutable_camers_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    double *mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * point_block_size();
    }

    const double *camera_for_observation(int i) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    const double *point_for_observation(int i) const {
        return points() + point_index_[i] * point_block_size();
    }


private:
    void CameraToAngleAxisAndCenter(const double *camera,
            double *angle_axis,
            double *center) const;

    void AngleAxisAndCenterToCamera(const double *angle_axis,
            const double *center,
            double *camera) const;

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    bool use_quaternions_;

    int *point_index_; // each observation corresponds to a point index
    int *camera_index_; // each observation correspoinds to a camera index
    double *observations_;
    double *parameters_;

};

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1)
        std::cerr << "Invalid UW data file. ";
}

void PerturbPoint3(const double sigma, double *point) {
    for (int i = 0; i < 3; ++i) {
        point[i] += RandNormal() * sigma;
    }
}

double Median(std::vector<double> *data) {
    int n = data->size();
    std::vector<double>::iterator mid_point = data->begin() + n / 2;
    std::nth_element(data->begin(), mid_point, data->end());
    return *mid_point;
}

BALProblem::BALProblem(const std::string &filename, bool use_quaternion) {
    FILE *fptr = fopen(filename.c_str(), "r");

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file." << filename;
        return;
    };

    /**
     * https://grail.cs.washington.edu/projects/bal/
     *
     * ----- Camera Model -----
     *
     * We use a pinhole camera model; the parameters we estimate for each camera area rotation R,
     * a translation t, a focal length f and two radial distortion parameters k1 and k2.
     * The formula for projecting a 3D point X into a camera R,t,f,k1,k2 is:
     *
     * P  =  R * X + t       (conversion from world to camera coordinates)
     * p  = -P / P.z         (perspective division)
     * p' =  f * r(p) * p    (conversion to pixel coordinates)
     *
     * where P.z is the third (z) coordinate of P. In the last equation,
     * r(p) is a function that computes a scaling factor to undo the radial distortion:
     * r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
     *
     * This gives a projection in pixels, where the origin of the image is the center of the image,
     * the positive x-axis points right, and the positive y-axis points up
     * (in addition, in the camera coordinate system, the positive z-axis points backwards,
     * so the camera is looking down the negative z-axis, as in OpenGL).
     *
     *
     * ----- Data Format -----
     *
     * Each problem is provided as a bzip2 compressed text file in the following format.
     * <num_cameras> <num_points> <num_observations>
     * <camera_index_1> <point_index_1> <x_1> <y_1>
     * ...
     * <camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
     * <camera_1>
     * ...
     * <camera_num_cameras>
     * <point_1>
     * ...
     * <point_num_points>
     *
     * Where, there camera and point indices start from 0. Each camera is a set of
     * 9 parameters - R,t,f,k1 and k2. The rotation R is specified as a Rodrigues' vector.
     */

    // This will die horribly on invalid files. Them's the breaks.
    FscanfOrDie(fptr, "%d", &num_cameras_); // 16 cameras, index
    FscanfOrDie(fptr, "%d", &num_points_); // 22106 points, index
    /**
     * There are totally 83718 observed point coordinates for each camera as the ground truth, z[x_1, y_1].
     * e.g.
     * <num_cameras> <num_points> <num_observations>
     * <camera_index_1> <point_index_1> <x_1> <y_1>
     * 0 0     -3.859900e+02 3.871200e+02
     * 1 0     -3.844000e+01 4.921200e+02
     * 2 0     -6.679200e+02 1.231100e+02
     * 7 0     -5.991800e+02 4.079300e+02
     * 12 0     -7.204300e+02 3.143400e+02
     * 13 0     -1.151300e+02 5.548999e+01
     * 0 1     3.838800e+02 -1.529999e+01
     * 1 1     5.597500e+02 -1.061500e+02
     * ...
     *
     *
     * <point_num_points>
     * -1.6943983532198115e-02
     * 1.1171804676513932e-02
     * 2.4643508831711991e-03
     * 7.3030995682610689e-01
     * -2.6490818471043420e-01
     * -1.7127892627337182e+00
     * 1.4300319432711681e+03
     * -7.5572758535864072e-08
     * 3.2377569465570913e-14
     * 1.5049725341485708e-02
     *
     * Although the first and second line has same point index,
     * they are different in different camera.
     *
     */
    FscanfOrDie(fptr, "%d", &num_observations_);

    std::cout << "Header: " << num_cameras_
              << " " << num_points_
              << " " << num_observations_ << std::endl;
    // Header: 16 22106 83718

    // Read data from BA.txt
    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    // all parameters needed to be optimized
    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    // If the data is read previously, it will never read this time,
    // so it will skip the first line.
    // [camera_i, point_, z_ij_x, z_ij_y]
    for (int i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", camera_index_ + i);
        FscanfOrDie(fptr, "%d", point_index_ + i);
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
        }
    }

    // optimizing variables
    for (int k = 0; k < num_parameters_; ++k) {
        FscanfOrDie(fptr, "%lf", parameters_ + k);
    }

    fclose(fptr);

    use_quaternions_ = use_quaternion;
    if (use_quaternion) {
        // Switch the angle-axis rotations to quaternions
        num_parameters_ = 10 * num_cameras_ + 3 * num_points_;
        double *quaternion_parameters = new double[num_parameters_];
        double *original_cursor = parameters_; // the first line of angle-axis
        double *quaternion_cursor = quaternion_parameters;
        for (int i = 0; i < num_cameras_; ++i) {
            AngleAxisToQuaternion(original_cursor, quaternion_cursor);
            quaternion_cursor += 4;
            original_cursor += 3;

            for (int j = 0; j < 10; ++j) {
                *quaternion_cursor++ = *original_cursor++;
            }
        }
        // Copy the rest of points
        for (int i = 0; i < 3 * num_points_; ++i) {
            *quaternion_cursor++ = *original_cursor++;
        }

        // Swap in the quaternion parameters
        delete[] parameters_;
        parameters_ = quaternion_parameters;
    }
}

void BALProblem::WriteToFile(const std::string &filename) const {
    FILE *fptr = fopen(filename.c_str(), "w");

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    }

    fprintf(fptr, "%d %d %d %d\n", num_cameras_, num_cameras_, num_points_, num_observations_);

    for (int i = 0; i < num_observations_; ++i) {
        fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
        for (int j = 0; j < 2; ++j) {
            fprintf(fptr, " %g", observations_[2 * i + j]);
        }
        fprintf(fptr, "\n");
    }

    for (int i = 0; i < num_cameras_; ++i) {
        double angleaxis[9];
        if (use_quaternions_) {
            // Output in angle-axis format.
            QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);
            memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double ));
        } else {
            memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
        }
        for (int j = 0; j < 9; ++i) {
            fprintf(fptr, "%.16g\n", angleaxis[j]);
        }
    }

    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            fprintf(fptr, "%.16g\n", point[j]);
        }
    }
    fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare
void BALProblem::WriteToPLYFile(const std::string &filename) const {
    std::ofstream of(filename.c_str(), std::ofstream::out);

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras_ + num_points_ // 16 + 22106
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header"
       << std::endl;

    /**
     * Export extrinsic data (i.e. camera centers in global coordinate) as green (0, 255, 0) points
     * 16 lines/cameras
     * -21.5216 -0.397644 169.909 0 255 0
     * -7.25499 0.560957 168.728 0 255 0
     * -81.9097 -37.8807 150.347 0 255 0
     * -93.391 -39.4746 153.356 0 255 0
     * -41.1762 -10.3846 162.287 0 255 0
     * ...
     */
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras_; ++i) {
        const double *camera = cameras() + camera_block_size() * i;
        CameraToAngleAxisAndCenter(camera, angle_axis, center);
        of << center[0] << ' ' << center[1] << ' ' << center[2] << " 0 255 0" << '\n';
    }

    /**
     * Export the structure (i.e. 3D Points in global coordinate) as white points.
     * -67.5344 41.6483 -52.4505  255 255 255
     * 24.4465 -12.7628 24.3263  255 255 255
     * 60.0435 3.10939 -5.03598  255 255 255
     * 60.3161 1.78752 -5.38407  255 255 255
     * 58.0433 84.0872 -103.763  255 255 255
     * -13.3213 -1.45887 31.9879  255 255 255
     * 27.2705 40.5469 -89.6207  255 255 255
     * ...
     */
    const double *points = parameters_ + camera_block_size() * num_cameras_; // 9 * 16 = 144
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) { // point_block_size = 3
            of << point[j] << ' ';
        }
        of << " 255 255 255\n";
    }
    of.close();
}

void BALProblem::CameraToAngleAxisAndCenter(const double *camera,
                                            double *angle_axis,
                                            double *center) const {
    VectorRef angle_axis_ref(angle_axis, 3); // Vector3d
    // needs angle-axis
    if (use_quaternions_) {
        QuaternionToAngleAxis(camera, angle_axis);
    } else {
        angle_axis_ref = ConstVectorRef(camera, 3);
    }
    // got angle-axis, camera pose

    // c = -R't
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    AngleAxisRotatePoint(inverse_rotation.data(),
            camera + camera_block_size() - 6, // 9 - 6 = 3
            center);
    VectorRef(center, 3) *= -1.0;
}

void BALProblem::AngleAxisAndCenterToCamera(const double *angle_axis,
                                            const double *center,
                                            double *camera) const {
    ConstVectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        AngleAxisToQuaternion(angle_axis, camera);
    } else {
        VectorRef(camera, 3) = angle_axis_ref;
    }

    // t = -R * t
    AngleAxisRotatePoint(angle_axis, center, camera + camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6, 3) *= -1.0;
}

void BALProblem::Normalize() {
    // compute the maginal median of the geometry
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;
    double *points = mutable_points();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < num_points_; ++j) {
            tmp[j] = points[3 * j + i];
        }
        median(i) = Median(&tmp);
    }

    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        tmp[i] = (point - median).lpNorm<1>();
    }

    const double median_absolute_deviation = Median(&tmp);

    // Scale so that the median absolute deviation of
    // the resulting reconstruction is 100
    const double scale = 100.0 / median_absolute_deviation;

    // X = scale * (X - median)
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        point = scale * (point - median);
    }

    double *cameras = mutable_cameras();
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = cameras + camera_block_size() * i;
        CameraToAngleAxisAndCenter(camera, angle_axis, center);
        // center = scale * (center - median)
        VectorRef (center, 3) = scale * (VectorRef(center, 3) - median);
        AngleAxisAndCenterToCamera(angle_axis, center, camera);
    }
}

// add random noise
void BALProblem::Perturb(const double rotation_sigma,
                         const double translation_sigma,
                         const double point_sigma) {
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    double *points = mutable_points();
    if (point_sigma > 0) {
        for (int i = 0; i < num_points_; ++i) {
            PerturbPoint3(point_sigma, points + 3 * i);
        }
    }

    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = mutable_cameras() + camera_block_size() * i;

        double angle_axis[3];
        double center[3];
        // Perturb in the rotation of the camera in the angle-axis representation
        CameraToAngleAxisAndCenter(camera, angle_axis, center);
        if (rotation_sigma > 0.0) {
            PerturbPoint3(rotation_sigma, angle_axis);
        }
        AngleAxisAndCenterToCamera(angle_axis, center, camera);

        if (translation_sigma > 0.0)
            PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
    }
}

#endif //COMMON_H