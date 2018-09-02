//
//  main.cpp
//  GeoHash
//
//  Created by Daniel Mesham on 13/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#include "edgy.hpp"
#include "geo_hash.h"
#include "hashing.hpp"
#include "lsq.hpp"
#include "models.hpp"
#include "orange.hpp"

#include <opencv2/core/core.hpp>
#include <iostream>
#include <chrono>

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
void debugShowBoxPoints(vector<Point2f> points, vector<vector<int>> edges, vector<bool> visibility);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * *


// The intrinsic matrix: Mac webcam
static float intrinsicMatrix[3][3] = {
    { 1047.7,    0  , 548.1 },
    {    0  , 1049.2, 362.9 },
    {    0  ,    0  ,   1   }
};
static Mat K = Mat(3,3, CV_32FC1, intrinsicMatrix);

static float binWidth = 1;
static int numBinsX = 12;
static float defaultZ = 500;

static string dataFolder = "../../../../../data/";



// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Main Method
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

int main(int argc, const char * argv[]) {
    
    // * * * * * * * * * * * * * * * * *
    //   HASHING
    // * * * * * * * * * * * * * * * * *
    
    auto startHash = chrono::system_clock::now(); // Start hashing timer
    
    vector<HashTable> tables;
    
    //Model * model = new Box(60, 80, 30);
    Model * model = new Rectangle(60, 80, Scalar(10, 45, 135));
    Mat modelMat = model->pointsToMat();
    vector<Point3f> modelPoints = model->getVertices();
    
    // A list of model basis pairs
    vector<vector<int>> basisList = model->getEdgeBasisList();
    
    
    float dA = 2 * CV_PI / numBinsX;    // Bin width
    
    for (int xBin = 0; xBin < numBinsX; xBin++) {
        float angleX = dA * (0.5 + xBin);
        for (int yBin = 0; yBin < numBinsX/2; yBin++) {
            float angleY = (dA * (0.5 + yBin)) - CV_PI/2;
            
            Vec6f pose = {0, 0, defaultZ, angleX, angleY, 0};
            Mat proj = lsq::projection(pose, modelMat, K);
            vector<Point2f> projPts = matToPoints(proj);
            vector<bool> vis = model->visibilityMask(angleX, angleY);

            for (int i = 0; i < basisList.size(); i++) {
                vector<int> basisIndex = basisList[i];
                
                // Don't make a hash table if the basis isn't visible
                if (!vis[basisIndex[0]] || !vis[basisIndex[1]]) continue;
                
                tables.push_back(hashing::createTable(basisIndex, projPts, vis, binWidth, {xBin, yBin}));
            }
        }
    }
    
    auto endHash = chrono::system_clock::now();
     
    // * * * * * * * * * * * * * * * * *
    //   CAMERA INPUT LOOP
    // * * * * * * * * * * * * * * * * *
    
    namedWindow("img", CV_WINDOW_AUTOSIZE);
    Mat img;
    vector<Vec4i> lines;
    int fileNum = 1;
    
    while(fileNum <= 9) {
        stringstream ss;
        ss << "orangeRect_" << fileNum << ".jpg";
        string s = ss.str();
        img = imread(dataFolder + ss.str(), CV_LOAD_IMAGE_COLOR);
        
        img = orange::segmentByColour(img, model->colour);
        
        // Get the detected lines
        vector<Vec4i> lines = orange::borderLines(img);
        
        // Stop if no lines found
        if (lines.size() <= 0) return 1;
        
        // TRACE: Display the lines on the images
        for(int i = 0; i < lines.size(); i++) {
            Vec4i l = lines[i];
            Point p1 = Point(l[0], l[1]);
            Point p2 = Point(l[2], l[3]);
            
            Scalar colour = Scalar(0,255,0);
            if (i == 0) colour = Scalar(255,0,0);
            if (i >= 4) colour = Scalar(0,0,255);
            
            line(img, p1, p2, colour, 1);
        }
        
        // Create the Mat of edge endpoints
        Mat target = edgy::edgeToPointsMat(lines[0]);
        for (int i = 1; i < lines.size(); i++) {
            Mat edgePts = edgy::edgeToPointsMat(lines[i]);
            hconcat(target, edgePts, target);
        }
        vector<Point2f> imgPoints = matToPoints(target);
        
        // Cluster the edge endpoints to get 4 corners
        vector<Point2f> clusterPts = edgy::clusterEdges(imgPoints, 4);
        for (int i = 0; i < clusterPts.size(); i++) {
            circle(img, Point(clusterPts[i]), 2, Scalar(120, 120, 255));
        }
        // Uncomment the line below to use the cluster centres as the corners
        //imgPoints = clusterPts;
        
        imshow("Original", img);
        
        // * * * * * * * * * * * * * *
        //      RECOGNITION
        // * * * * * * * * * * * * * *
        
        auto startRecog = chrono::system_clock::now(); // Start recognition timer
        auto endRecog1 = startRecog;                    // Placeholder for the first end time
        
        vector<estimate> estList;       // List if pose estimates
        int edge = 0;                   // The detected edge to use as a basis
        
        while (edge < lines.size() && estList.size() < 1) {
            //TRACE:
            cout << endl << "TRYING EDGE #" << edge << endl;
            
            vector<int> imgBasis = {2*edge, 2*edge +1};

            vector<HashTable> votedTables = hashing::voteForTables(tables, imgPoints, imgBasis);

            // Use least squares to match the tables with the most votes
            int maxVotes = votedTables[0].votes;
            cout << "MAX VOTES = " << maxVotes << endl << endl;
            
            
            
            for (int i = 0; i < votedTables.size(); i++) {
                HashTable t = votedTables[i];
                if (t.votes < MIN(200, maxVotes)) break;
                
                vector<Mat> orderedPoints = hashing::getOrderedPoints(imgBasis, t, modelPoints, imgPoints);
                
                Mat newModel = orderedPoints[0];
                Mat newTarget = orderedPoints[1];
                
                // Take only 4 correspondences
                if (newModel.cols <= 0) continue;
                if (newModel.cols > 4) {
                    newModel = newModel.colRange(0, 4);
                    newTarget = newTarget.rowRange(0, 4);
                }
                else if (newModel.cols < 4) {
                    //TRACE
                    cout << "* * * Only " << newModel.cols << " correspondences!!" << endl;
                }
                
                float xAngle = dA * (0.5 + t.viewAngle[0]);
                float yAngle = (dA * (0.5 + t.viewAngle[1])) - CV_PI/2;
                Vec6f poseInit = {0, 0, defaultZ, xAngle, yAngle, 0};
                estimate est = lsq::poseEstimateLM(poseInit, newModel, newTarget, K);
                
                if (est.iterations != lsq::MAX_ITERATIONS && est.pose[2] > 0 && !isnan(est.error)) {
                    //TRACE
                    cout << "Basis = " << t.basis[0] << "," << t.basis[1] << " | Angle = " << t.viewAngle[0] << "," << t.viewAngle[1] << " | Votes = " << t.votes <<  endl;
                
                    est.print();
                    estList.push_back(est);
                    if (estList.size() == 1) endRecog1 = chrono::system_clock::now();
                }
            }
            edge++;
        }
        
        auto endRecog = chrono::system_clock::now();
        
        chrono::duration<double> timeHash = endHash-startHash;
        cout << "Hashing time     = " << timeHash.count()*1000.0 << " ms" << endl;
        chrono::duration<double> timeRecog = endRecog-startRecog;
        cout << "Recognition time = " << timeRecog.count()*1000.0 << " ms" << endl;
        chrono::duration<double> timeRecog1 = endRecog1-startRecog;
        cout << "First match time = " << timeRecog1.count()*1000.0 << " ms" << endl;
        
        cout << endl << estList.size() << " successes!" << endl;
        
        // TRACE
        if (estList.size() > 0) {
            sort(estList.begin(), estList.end());
            Mat imgResult;
            img.copyTo(imgResult);
            model->draw(imgResult, estList[0].pose, K, Scalar(0,0,255));
            imshow("imgResult", imgResult);
            cout << "\nSmallest error = \n"; estList[0].print();
        }

        if (waitKey(0) == 'n') fileNum++;
        
    }
    
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

void debugShowBoxPoints(vector<Point2f> points, vector<vector<int>> edges, vector<bool> visibility) {
    // Show the visibile points on the image
    Mat img = Mat::zeros(900, 1200, CV_32FC3);
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
