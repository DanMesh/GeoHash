//
//  models.cpp
//  GeoHash
//
//  Created by Daniel Mesham on 16/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "models.hpp"



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

const vector<vector<int>> Box::edgeBasisList = {
    {0,1}, {1,2}, {2,3}, {3,0},
    {1,0}, {2,1}, {3,2}, {0,3},
    
    {4,5}, {5,6}, {6,7}, {7,4},
    {5,4}, {6,5}, {7,6}, {4,7},
    
    {0,4}, {1,5}, {2,6}, {3,7},
    {4,0}, {5,1}, {6,2}, {7,3}
};

bool Box::vertexIsVisible(int vertexID, float xAngle, float yAngle) {
    while (xAngle < 0)          xAngle += 2*CV_PI;
    while (xAngle >= 2*CV_PI)   xAngle -= 2*CV_PI;
    
    while (yAngle < 0)          yAngle += 2*CV_PI;
    while (yAngle >= 2*CV_PI)   yAngle -= 2*CV_PI;
    if (yAngle > 0.5*CV_PI && yAngle < 1.5*CV_PI) throw invalid_argument("Bad yAngle for visibility check. yAngle must be between ±PI/2 (or within a 2*PI difference)");
    
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
