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
    double res_rho = 1;
    double res_theta = CV_PI/180;
    int threshold = 30;
    double minLineLength = 5;
    double maxLineGap = 10;
    
    // Edge detection
    Mat dst;
    Canny(img, dst, 40, 120);
    
    // Line detection
    vector<Vec4i> lines;
    while ((lines.size() < 3) && (threshold > 0)) {
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

void orange::getCorners(Mat img) {
    //TODO
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
