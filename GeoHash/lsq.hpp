//
//  lsq.hpp
//  LiveTracker
//
//  Created by Daniel Mesham on 07/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef lsq_hpp
#define lsq_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      A class defining a least squares estimated pose
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
class estimate {
public:
    estimate( Vec6f pose_in, float error_in, float iter_in ) : pose(pose_in), error(error_in), iterations(iter_in) {}
    void print() {cout << pose << "\nIterations = " << iterations << "\nError = " << error << "\n\n";}
public:
    Vec6f pose;
    float error, iterations;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      A library of least squares methods
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
class lsq {
    
/*
    METHODS
 */
public:
    static estimate poseEstimateLM(Vec6f pose1, Mat x, Mat target, Mat K);
    static Mat translation(float x, float y, float z);
    static Mat rotation(float x, float y, float z);
    static Mat projection(Vec6f pose, Mat x, Mat K);
    static float projectionError(Mat target, Mat proj);
    static Mat pointsAsCol(Mat points);
    static Mat jacobian(Vec6f pose, Mat x, Mat K);
    
/*
    CONSTANTS
 */
public:
    static const int MAX_ITERATIONS = 20;

};




#endif /* lsq_hpp */
