//
//  lsq.cpp
//  LiveTracker
//
//  Created by Daniel Mesham on 07/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "lsq.hpp"

const float lsq::ERROR_THRESHOLD = 0.5;

estimate lsq::poseEstimateLM(Vec6f pose1, Mat model, Mat target, Mat K) {
    // pose1: imitial pose parameters
    // model: model points in full homogeneous coords
    // target: image points, in 2D coords
    // K: intrinsic matrix
        
    Mat y1 = lsq::projection(pose1, model, K);
    float E = lsq::projectionError(target, y1);
    
    float lambda = 0;
    
    int iterations = 1;
    while (E > ERROR_THRESHOLD && iterations < MAX_ITERATIONS) {
        Mat J = lsq::jacobian(pose1, model, K);
        Mat eps;
        subtract(lsq::projection(pose1, model, K).rowRange(0, 2).t(), target, eps);
        eps = lsq::pointsAsCol(eps.t());
        Mat Jp = J.t() * J + lambda * Mat::eye(6,6,CV_32FC1);
        Jp = -Jp.inv() * J.t();
        Mat del = Jp * eps;
        
        Vec6f pose2 = pose1;
        for (int i = 0; i < 6; i++) {
            pose2[i] += del.at<float>(i);
        }
        
        Mat y2 = lsq::projection(pose2, model, K);
        float E2 = lsq::projectionError(target, y2);
        
        pose1 = pose2;
        E = E2;
        
        iterations++;
    }
    
    return estimate(pose1, E, iterations);
}

Mat lsq::translation(float x, float y, float z) {
    // Translate by the given x, y and z values
    float tmp[] = {x, y, z};
    return Mat(3, 1, CV_32FC1, tmp) * 1;
}

Mat lsq::rotation(float x, float y, float z) {
    // Rotate about the x, y then z axes with the given angles in radians
    float rotX[3][3] = {
        { 1,       0,       0 },
        { 0,  cos(x), -sin(x) },
        { 0,  sin(x),  cos(x) }
    };
    float rotY[3][3] = {
        {  cos(y),   0,  sin(y) },
        {       0,   1,       0 },
        { -sin(y),   0,  cos(y) }
    };
    float rotZ[3][3] = {
        {  cos(z), -sin(z),  0 },
        {  sin(z),  cos(z),  0 },
        {       0,       0,  1 }
    };
    
    Mat rX = Mat(3, 3, CV_32FC1, rotX);
    Mat rY = Mat(3, 3, CV_32FC1, rotY);
    Mat rZ = Mat(3, 3, CV_32FC1, rotZ);
    
    return rZ * rY * rX;
}

Mat lsq::projection(Vec6f pose, Mat model, Mat K) {
    Mat P;
    hconcat( rotation(pose[3], pose[4], pose[5]) , translation(pose[0], pose[1], pose[2]) , P);
    Mat y = (K * P) * model;
    
    Mat z = y.row(2);
    Mat norm;
    vconcat(z, z, norm);
    vconcat(norm, z, norm);
    divide(y, norm, y);
    return y;
}

float lsq::projectionError(Mat target, Mat proj) {
    int numPoints = proj.cols;
    transpose(proj.rowRange(0, 2), proj);
    
    Mat e;
    subtract(target, proj, e);
    multiply(e, e, e);
    reduce(e, e, 1, CV_REDUCE_SUM, CV_32FC1);
    sqrt(e, e);
    Mat eT;
    transpose(e, eT);
    e = eT * e;
    return e.at<float>(0) / numPoints;
}

Mat lsq::pointsAsCol(Mat points) {
    // Converts a matrix of points (each a column vector in homogeneous coordinates) into a single column of coordinates in standard coordinates (i.e. the last coordinate is removed)
    points = points.rowRange(0, 2);
    points = points.t();
    points = points.reshape(0, 1).t();
    return points;
}

Mat lsq::jacobian(Vec6f pose, Mat model, Mat K) {
    // Calculates the Jacobian for the given pose of model x
    Mat J = Mat(2*model.cols, 0, CV_32FC1);
    
    float dt = 0.5;
    float dr = CV_PI/360;
    vector<float> delta = {dt, dt, dt, dr, dr, dr};
    
    for (int i = 0; i < 6; i++) {
        Vec6f p1 = pose;
        p1[i] += delta[i];
        Vec6f p2 = pose;
        p2[i] -= delta[i];
        Mat j = (projection(p1, model, K) - projection(p2, model, K))/delta[i];
        hconcat(J, pointsAsCol(j), J);
    }
    
    return J;
}

Vec6f estimate::standardisePose(Vec6f pose) {
    // Ensures all angles are in (-PI,PI]
    for (int i = 3; i < 6; i++) {
        while (pose[i] > CV_PI)     pose[i] -= CV_2PI;
        while (pose[i] <= -CV_PI)   pose[i] += CV_2PI;
    }
    return pose;
}
