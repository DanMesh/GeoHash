//
//  models.cpp
//  GeoHash
//
//  Created by Daniel Mesham on 16/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "lsq.hpp"
#include "models.hpp"


// * * * * * * * * * * * * * * *
//      Box
// * * * * * * * * * * * * * * *

const vector<vector<float>> Box::xAngleLimits = {
    {1.0*CV_PI, 1.5*CV_PI}, {1.0*CV_PI, 1.5*CV_PI},
    {0.5*CV_PI, 1.0*CV_PI}, {0.5*CV_PI, 1.0*CV_PI},
    {1.5*CV_PI, 2.0*CV_PI}, {1.5*CV_PI, 2.0*CV_PI},
    {0.0*CV_PI, 0.5*CV_PI}, {0.0*CV_PI, 0.5*CV_PI}
};
const vector<vector<float>> Box::yAngleLimits = {
    {0.0*CV_PI, 0.5*CV_PI}, {1.5*CV_PI, 2.0*CV_PI},
    {1.5*CV_PI, 2.0*CV_PI}, {0.0*CV_PI, 0.5*CV_PI},
    {0.0*CV_PI, 0.5*CV_PI}, {1.5*CV_PI, 2.0*CV_PI},
    {1.5*CV_PI, 2.0*CV_PI}, {0.0*CV_PI, 0.5*CV_PI}
};

const vector<vector<int>> Box::faces = {
    {0,1,2,3}, {0,1,5,4}, {0,3,7,4},
    {4,5,6,7}, {1,2,6,5}, {2,3,7,6}
};

bool Box::vertexIsVisible(int vertexID, float xAngle, float yAngle) {
    while (yAngle < 0)          yAngle += 2*CV_PI;
    while (yAngle >= 2*CV_PI)   yAngle -= 2*CV_PI;
    if (yAngle > 0.5*CV_PI && yAngle < 1.5*CV_PI) {
        yAngle = CV_PI - yAngle;            // Bring into band around 0 radians
        if (yAngle < 0) yAngle += 2*CV_PI;  // Make angle positive again
        xAngle += CV_PI;                    // Add PI to the xAngle
    }
    
    while (xAngle < 0)          xAngle += 2*CV_PI;
    while (xAngle >= 2*CV_PI)   xAngle -= 2*CV_PI;
    
    vector<float> xLim = xAngleLimits[vertexID];
    vector<float> yLim = yAngleLimits[vertexID];
    if (xAngle >= xLim[0] && xAngle <= xLim[1]) {
        if (yAngle >= yLim[0] && yAngle <= yLim[1]) {
            return false;
        }
    }
    return true;
}

vector<bool> Box::visibilityMask(float xAngle, float yAngle) {
    vector<bool> mask;
    for (int i = 0; i < 8; i++) {
        mask.push_back(vertexIsVisible(i, xAngle, yAngle));
    }
    return mask;
}

void Box::createPoints(float w, float h, float d) {
    w = w/2;
    h = h/2;
    d = d/2;
    Point3f p0 = Point3f(-w, -h, -d);
    Point3f p1 = Point3f( w, -h, -d);
    Point3f p2 = Point3f( w,  h, -d);
    Point3f p3 = Point3f(-w,  h, -d);
    Point3f p4 = Point3f(-w, -h,  d);
    Point3f p5 = Point3f( w, -h,  d);
    Point3f p6 = Point3f( w,  h,  d);
    Point3f p7 = Point3f(-w,  h,  d);
    vertices = {p0, p1, p2, p3, p4, p5, p6, p7};
}

Mat Box::pointsToMat() {
    Mat ret = Mat(4, int(vertices.size()), CV_32FC1);
    for (int i = 0; i < vertices.size(); i++) {
        Point3f p = vertices[i];
        ret.at<float>(0, i) = p.x;
        ret.at<float>(1, i) = p.y;
        ret.at<float>(2, i) = p.z;
        ret.at<float>(3, i) = 1;
    }
    return ret * 1;
}

void Box::draw(Mat img, Vec6f pose, Mat K, bool lines, Scalar colour) {
    Mat proj = lsq::projection(pose, pointsToMat(), K);
    
    // Create a list of points
    vector<Point> points;
    for (int i  = 0; i < proj.cols; i++) {
        Mat col = proj.col(i);
        points.push_back(Point(col.at<float>(0), col.at<float>(1)));
    }
    
    // Draw the points according to the edge list
    for (int i = 0; i < faces.size(); i++) {
        vector<int> face = faces[i];
        
        if (!vertexIsVisible(face[0], pose[3], pose[4])) continue; // Don't show invisible vertices
        if (!vertexIsVisible(face[1], pose[3], pose[4])) continue;
        if (!vertexIsVisible(face[2], pose[3], pose[4])) continue;
        if (!vertexIsVisible(face[3], pose[3], pose[4])) continue;
        
        Point p1 = points[ face[0] ];
        Point p2 = points[ face[1] ];
        Point p3 = points[ face[2] ];
        Point p4 = points[ face[3] ];
        Point pts[1][4] = {
            {points[ face[0] ], points[ face[1] ], points[ face[2] ], points[ face[3] ]}
        };
        const Point* ppt[1] = {pts[0]};
        int npt[] = {4};
        
        if (lines) polylines(img, ppt, npt, 1, true, colour, 1);
        else fillPoly(img, ppt, npt, 1, colour*(1 - i*0.1));
    }
}


// * * * * * * * * * * * * * * *
//      Rectangle
// * * * * * * * * * * * * * * *

bool Rectangle::vertexIsVisible(int vertexID, float xAngle, float yAngle) {
    return true;
}

vector<bool> Rectangle::visibilityMask(float xAngle, float yAngle) {
    return {true, true, true, true};
}

void Rectangle::createPoints(float w, float h) {
    w = w/2;
    h = h/2;
    Point3f p0 = Point3f(-w, -h, 0);
    Point3f p1 = Point3f( w, -h, 0);
    Point3f p2 = Point3f( w,  h, 0);
    Point3f p3 = Point3f(-w,  h, 0);
    vertices = {p0, p1, p2, p3};
}

Mat Rectangle::pointsToMat() {
    Mat ret = Mat(4, int(vertices.size()), CV_32FC1);
    for (int i = 0; i < vertices.size(); i++) {
        Point3f p = vertices[i];
        ret.at<float>(0, i) = p.x;
        ret.at<float>(1, i) = p.y;
        ret.at<float>(2, i) = p.z;
        ret.at<float>(3, i) = 1;
    }
    return ret * 1;
}
