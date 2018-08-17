//
//  main.cpp
//  GeoHash
//
//  Created by Daniel Mesham on 13/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "geo_hash.h"
#include "hashing.hpp"
#include "lsq.hpp"
#include "models.hpp"

#include <opencv2/core/core.hpp>
#include <iostream>
#include <map>

using namespace std;
using namespace cv;


// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Function Prototypes
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Point2f basisCoords(vector<Point2f> basis, Point2f p);
vector<Point2f> matToPoints(Mat points);
Mat pointsToMat(vector<Point2f> points);
Mat reformModelMat(vector<Point2f> modelPoints2D);
vector<vector<int>> createBasisList(int numPoints);
bool vectorContains(vector<int> vec, int query);
void debugShowBoxPoints(vector<Point2f> points, vector<bool> visibility);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * *


// The intrinsic matrix: Mac webcam
static float intrinsicMatrix[3][3] = {
    { 1047.7,    0  , 548.1 },
    {    0  , 1049.2, 362.9 },
    {    0  ,    0  ,   1   }
};
static Mat K = Mat(3,3, CV_32FC1, intrinsicMatrix);

// The points of the model
static float rectModel[4][4] = {
    { 0,  60,  60,   0},
    { 0,   0,  80,  80},
    { 0,   0,   0,   0},
    { 1,   1,   1,   1}
};
static Mat x = Mat(4,4, CV_32FC1, rectModel);

static float binWidth = 2;
static int numBinsX = 12;
static float defaultZ = 500;


int main(int argc, const char * argv[]) {
    
    
    vector<HashTable> tables;
    //vector<Point2f> modelPoints = matToPoints(x);
    
    
    Box modelBox = Box(60, 80, 30);
    Mat modelMat = modelBox.pointsToMat();
    vector<Point3f> modelPoints = modelBox.vertices;
    
    // A list of model basis pairs
    vector<vector<int>> basisList = modelBox.edgeBasisList;
    
    
    float dA = 2 * CV_PI / numBinsX;    // Bin width
    
    for (int xBin = 0; xBin < numBinsX; xBin++) {
        float angleX = dA * (0.5 + xBin);
        for (int yBin = 0; yBin < numBinsX/2; yBin++) {
            float angleY = (dA * (0.5 + yBin)) - CV_PI/2;
            
            Vec6f pose = {0, 0, defaultZ, angleX, angleY, 0};
            Mat proj = lsq::projection(pose, modelMat, K);
            vector<Point2f> projPts = matToPoints(proj);
            vector<bool> vis = modelBox.visibilityMask(angleX, angleY);
            
            //TRACE
            debugShowBoxPoints(projPts, vis);
            
            cout << "Hashing...";
            for (int i = 0; i < basisList.size(); i++) {
                vector<int> basisIndex = basisList[i];
                
                // Don't make a hash table if the basis isn't visible
                if (!vis[basisIndex[0]] || !vis[basisIndex[1]]) continue;
                
                tables.push_back(hashing::createTable(basisIndex, projPts, vis, binWidth, {xBin, yBin}));
            }
            cout << "Done" << endl;
        }
    }
    
    
     
    // * * * * * * * * * * * * * * * * *
    //   Create a set of image points
    // * * * * * * * * * * * * * * * * *
    
    float rX = CV_PI/3;
    float rY = -CV_PI/4;
    
    Vec6f pose = {30, 12, 500, rX, rY, 0};
    Mat img = lsq::projection(pose, modelMat, K);
    vector<Point2f> imgPointsAll = matToPoints(img);
    vector<Point2f> imgPoints;
    vector<bool> visMask = Box::visibilityMask(rX, rY);
    
    for (int i = 0; i < imgPointsAll.size(); i++) {
        if (visMask[i]) imgPoints.push_back(imgPointsAll[i]);
    }
    
    Point2f noisePoint = Point2f(550,550);
    //imgPoints.push_back(noisePoint);
    
    cout << "MODEL = " << endl << modelPoints << endl << endl << "IMAGE = " << endl << imgPoints << endl << endl;
    
    debugShowBoxPoints(imgPoints, visMask);
    //waitKey(0);
    
    // * * * * * * * * * * * * * *
    //      RECOGNITION
    // * * * * * * * * * * * * * *
    
    vector<int> imgBasis = {0,1};    // The "random" basis

    vector<HashTable> votedTables = hashing::voteForTables(tables, imgPoints, imgBasis);

    // Use least squares to match the tables with the most votes
    int maxVotes = votedTables[0].votes;
    cout << "MAX VOTES = " << maxVotes << endl << endl;
    
    int successes = 0;
    
    for (int i = 0; i < votedTables.size(); i++) {
        HashTable t = votedTables[i];
        if (t.votes < MIN(100, maxVotes)) break;
        
        vector<Mat> orderedPoints = hashing::getOrderedPoints(imgBasis, t, modelPoints, imgPoints);
        
        Mat newModel = orderedPoints[0];
        Mat newTarget = orderedPoints[1];
        
        float xAngle = dA * (0.5 + t.viewAngle[0]);
        float yAngle = (dA * (0.5 + t.viewAngle[1])) - CV_PI/2;
        Vec6f poseInit = {0, 0, defaultZ, xAngle, yAngle, 0};
        estimate est = lsq::poseEstimateLM(poseInit, newModel, newTarget, K);
        
        if (est.iterations != lsq::MAX_ITERATIONS) {
            //TRACE
            cout << "Basis = " << t.basis[0] << "," << t.basis[1] << " | Angle = " << t.viewAngle[0] << "," << t.viewAngle[1] << " | Votes = " << t.votes <<  endl;
            
            est.print();
            successes++;
        }
    }
    
    cout << endl << successes << "/" << votedTables.size() << " successes!" << endl;

    return 0;
}

Point2f basisCoords(vector<Point2f> basis, Point2f p) {
    // Converts the coordinates of point p into the reference frame with the given basis
    Point2f O = (basis[0] + basis[1])/2;
    basis[0] -= O;
    basis[1] -= O;
    p = p - O;
    
    float B = sqrt(pow(basis[1].x, 2) + pow(basis[1].y, 2));
    float co = basis[1].x / B;
    float si = basis[1].y / B;
    
    float u =  co * p.x + si * p.y;
    float v = -si * p.x + co * p.y;
    
    return Point2f(u, v)/B;
}

vector<Point2f> matToPoints(Mat points) {
    vector<Point2f> ret;
    for (int i  = 0; i < points.cols; i++) {
        Mat col = points.col(i);
        ret.push_back(Point2f(col.at<float>(0), col.at<float>(1)));
    }
    return ret;
}

Mat pointsToMat(vector<Point2f> points) {
    int rows = 2;
    int cols = int(points.size());
    
    float table[rows][cols];
    for (int c = 0; c < cols; c++) {
        table[0][c] = points[c].x;
        table[1][c] = points[c].y;
    }
    return Mat(rows, cols, CV_32FC1, table) * 1;
}

Mat reformModelMat(vector<Point2f> modelPoints2D) {
    // Converts 2D model points into their full 3D homogeneous representation
    Mat m = pointsToMat(modelPoints2D);
    
    Mat zer = Mat::zeros(1, m.cols, CV_32FC1);
    Mat one = Mat::ones(1, m.cols, CV_32FC1);
    vconcat(m, zer, m);
    vconcat(m, one, m);
    return m * 1;
}

vector<vector<int>> createBasisList(int numPoints) {
    vector<vector<int>> ret;
    for (int i = 0; i < numPoints; i++) {
        for (int j = 0; j < numPoints; j++) {
            if (i != j) {
                ret.push_back({i, j});
            }
        }
    }
    return ret;
}

bool vectorContains(vector<int> vec, int query) {
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == query) return true;
    }
    return false;
}

void debugShowBoxPoints(vector<Point2f> points, vector<bool> visibility) {
    // Show the visibile points on the image
    Mat img = Mat::zeros(900, 1200, CV_32FC3);
    vector<vector<int>> edges = Box::edgeBasisList;
    for (int i = 0; i < edges.size(); i++) {
        int id1 = edges[i][0];
        int id2 = edges[i][1];
        if (!visibility[id1] || !visibility[id2]) continue;
        
        Point p1 = Point(points[id1]);
        Point p2 = Point(points[id2]);
        line(img, p1, p2, Scalar(0,0,255));
    }

    for (int i = 0; i < visibility.size(); i++) {
        cout << visibility[i] << " ";
        putText(img, to_string(i) , Point(points[i]), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255,255,250));
    }
    cout << endl;
    
    namedWindow( "Debug", CV_WINDOW_AUTOSIZE );
    imshow("Debug",img);
    //waitKey(0);
}
