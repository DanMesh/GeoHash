//
//  orange.cpp
//  LiveTracker
//
//  Created by Daniel Mesham on 01/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "orange.hpp"

using namespace std;
using namespace cv;


vector<Point> orange::getOrangeCorners(Mat img) {
    Mat planes[3];
    split(img, planes);
    Mat red = planes[2];
    
    vector<Vec4i> lines = borderLines(img);
    vector<vector<Point>> sides;
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        
        Point p1 = Point(l[0], l[1]);
        Point p2 = Point(l[2], l[3]);
        sides.push_back({p1, p2});
        
        line(img, p1, p2, Scalar(0,255,0), 1);
    }
        
    // Sort the sides
    sides = sortSides(sides);
    for (int i = 0; i < sides.size(); i++) {
        vector<Point> s = sides[i];
        cout << "[ " << s[0].x << ", " << s[0].y << " ]   [ " << s[1].x << ", " << s[1].y << " ]\n";
    }
    
    // Define the shape as a vector of vertices
    vector<Point> shape;
    shape.push_back(intersection(sides[sides.size() - 1], sides[0]));
    for (int i = 1; i < sides.size(); i++) {
        shape.push_back(intersection(sides[i - 1], sides[i]));
    }
    
    return shape;
}




vector<Vec4i> orange::borderLines(Mat img) {
    // Returns the lines corresponding to the edges of the rectangle from an image.
    
    // Resolutions of the rho and theta parameters of the lines
    double res_rho = 5;
    double res_theta = CV_PI/180;
    int threshold = 60;
    double minLineLength = 50;
    double maxLineGap = 30;
    
    // Edge detection
    Mat dst;
    Canny(img, dst, 70, 210, 3, true);
    
    // Line detection
    vector<Vec4i> lines;
    while ((lines.size() < 4) && (threshold > 0)) {
        HoughLinesP(dst, lines, res_rho, res_theta, threshold, minLineLength, maxLineGap);
        threshold -= 5;
    }
    
    return lines;
}

vector<vector<Point>> orange::sortSides(vector<vector<Point>> sides) {
    for (int i = 0; i < sides.size()-1; i++) {
        
        vector<Point> s = sides[i];
        vector<int> closest = {1, 0};
        double minDist = 10000;
        
        for (int j = i+1; j < sides.size(); j++) {
            for (int k = 0; k < 2; k++) {
                double dist = norm(s[1] - sides[j][k]);
                if (dist < minDist) {
                    minDist = dist;
                    closest = {j, k};
                }
            }
        }
        
        // Move the closest side to be immediately after the current side
        if (closest[0] != i+1) {
            vector<Point> tmp = sides[i+1];
            sides[i+1] = sides[closest[0]];
            sides[closest[0]] = tmp;
        }
        // Move the closest point to be the first in the side
        if (closest[1] != 0) {
            Point tmp = sides[i+1][0];
            sides[i+1][0] = sides[i+1][1];
            sides[i+1][1] = tmp;
        }
    }
    return sides;
}

/*
 * Returns the point of intersection of the lines defined by the vectors of points a and b.
 */
Point orange::intersection(vector<Point> a, vector<Point> b) {
    int x1 = a[0].x;
    int x2 = a[1].x;
    int x3 = b[0].x;
    int x4 = b[1].x;
    
    int y1 = a[0].y;
    int y2 = a[1].y;
    int y3 = b[0].y;
    int y4 = b[1].y;
    
    double x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4));
    double y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4));
    
    return Point(x, y);
}

vector<Point> orange::linesToCorners(vector<Vec4i> lines) {
    vector<vector<Point>> sides;
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        
        Point p1 = Point(l[0], l[1]);
        Point p2 = Point(l[2], l[3]);
        sides.push_back({p1, p2});
    }
    
    // Sort the sides
    sides = sortSides(sides);
    
    // Define the shape as a vector of vertices
    vector<Point> shape;
    shape.push_back(intersection(sides[sides.size() - 1], sides[0]));
    for (int i = 1; i < sides.size(); i++) {
        shape.push_back(intersection(sides[i - 1], sides[i]));
    }
    
    return shape;
}

Mat orange::segmentByColour(Mat img, Scalar colour) {
    
    // Convert colour to HSV
    Mat bgr(1 ,1 , CV_8UC3, colour);
    Mat3b hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    Vec3b hsvPixel(hsv.at<Vec3b>(0,0));
    
    // Establish H, S, V ranges
    int thr[3] = {30, 50, 50};
    Scalar minHSV = Scalar(hsvPixel.val[0] - thr[0], hsvPixel.val[1] - thr[1], hsvPixel.val[2] - thr[2]);
    Scalar maxHSV = Scalar(hsvPixel.val[0] + thr[0], hsvPixel.val[1] + thr[1], hsvPixel.val[2] + thr[2]);
    
    // * * * * * * * * * *
    //      Blur & Sharpen
    // * * * * * * * * * *
    
    Mat blurred;
    int k = 3;
    GaussianBlur(img, blurred, Size(k,k), 1);
    
    // * * * * * * * * * *
    //      Segment
    // * * * * * * * * * *
    
    Mat imgHSV, imgMask, imgResult;
    cvtColor(blurred, imgHSV, COLOR_BGR2HSV);
    inRange(imgHSV, minHSV, maxHSV, imgMask);
    bitwise_and(img, img, imgResult, imgMask);
    
    //return imgResult;
    
    // * * * * * * * * * *
    //      Open/Close
    // * * * * * * * * * *
    
    k = 1;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2*k + 1, 2*k + 1), Point(k, k));
    morphologyEx(imgResult, imgResult, MORPH_OPEN, kernel, Point(-1, -1), 1);
    morphologyEx(imgResult, imgResult, MORPH_CLOSE, kernel, Point(-1, -1), 1);
    
    return imgResult;
}
