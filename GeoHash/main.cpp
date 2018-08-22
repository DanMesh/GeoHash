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


static float binWidth = 2;
static int numBinsX = 12;
static float defaultZ = 500;



// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Main Method
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

int main(int argc, const char * argv[]) {
    
    VideoCapture cap(0); // Open the default camera
    if(!cap.isOpened())  // Check if we succeeded
        return -1;
    
    // Segmentation by colour
    // Using HSV (Hue, Saturation, Brightness)
    // From https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
    Mat3b bgr (Vec3b(25, 95, 215)); // Orange for rectangle
    //Mat3b bgr (Vec3b(55, 105, 20)); // Green for box
    
    Mat3b hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    Vec3b hsvPixel(hsv.at<Vec3b>(0,0));
    
    int thr[3] = {20, 50, 50};
    Scalar minHSV = Scalar(hsvPixel.val[0] - thr[0], hsvPixel.val[1] - thr[1], hsvPixel.val[2] - thr[2]);
    Scalar maxHSV = Scalar(hsvPixel.val[0] + thr[0], hsvPixel.val[1] + thr[1], hsvPixel.val[2] + thr[2]);
    
    // * * * * * * * * * * * * * * * * *
    //   HASHING
    // * * * * * * * * * * * * * * * * *
    
    auto startHash = chrono::system_clock::now(); // Start hashing timer
    
    vector<HashTable> tables;
    
    //Model * model = new Box(175, 210, 44);
    Model * model = new Rectangle(60, 80);
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
    
    namedWindow("imgResult", CV_WINDOW_AUTOSIZE);
    namedWindow("img", CV_WINDOW_AUTOSIZE);
    Mat img, imgLAB, imgHSV, imgResult;
    vector<Vec4i> lines;
    while(1) {
        cap >> img;
        
        Mat imgMask;
        imgResult = Mat();
        
        
        Mat kernelSharp = (Mat_<float>(3,3) <<
                      1,  1, 1,
                      1, -8, 1,
                      1,  1, 1);
        Mat imgLaplacian;
        Mat sharp = img; // copy source image to another temporary one
        filter2D(sharp, imgLaplacian, CV_32F, kernelSharp);
        img.convertTo(sharp, CV_32F);
        Mat imgSharp = sharp - imgLaplacian;
        // convert back to 8bits gray scale
        imgSharp.convertTo(imgSharp, CV_8UC3);
        imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
        //imshow( "Laplace Filtered Image", imgLaplacian );
        //imshow( "New Sharped Image", imgSharp );
        
        
        
        
        // Segmentation using the HSV color
        cvtColor(img, imgHSV, COLOR_BGR2HSV);
        inRange(imgHSV, minHSV, maxHSV, imgMask);
        bitwise_and(img, img, imgResult, imgMask);
        
        Mat planes[3], masks[3];
        split(imgHSV, planes);
        inRange(planes[0], minHSV[0], maxHSV[0], masks[0]);
        inRange(planes[1], minHSV[1], maxHSV[1], masks[1]);
        inRange(planes[2], minHSV[2], maxHSV[2], masks[2]);
        //imshow( "Hue in Range", masks[0] );
        //imshow( "Sat in Range", masks[1] );
        //imshow( "Int in Range", masks[2] );
        //imshow( "Hue", planes[2] );

        //imshow( "Thresholded Image", imgMask );
        
        
        // Blur the segmented image
        /*Mat blurred;
        GaussianBlur(imgResult, blurred, Size(0,0), 3);
        addWeighted(imgResult, 1.5, blurred, -0.5, 0, imgResult);*/
        
        // Use opening and closing for noise removal
        int k = 3;
        Mat kernel = getStructuringElement(MORPH_RECT, Size(2*k + 1, 2*k + 1), Point(k, k));
        morphologyEx(imgResult, imgResult, MORPH_OPEN, kernel, Point(-1, -1), 1);
        morphologyEx(imgResult, imgResult, MORPH_CLOSE, kernel, Point(-1, -1), 1);
        morphologyEx(imgMask, imgMask, MORPH_OPEN, kernel, Point(-1, -1), 2);
        morphologyEx(imgMask, imgMask, MORPH_CLOSE, kernel, Point(-1, -1), 2);
        
        imshow( "Open/Closed Image", imgMask );
        
        // Perform the distance transform algorithm
        Mat dist;
        distanceTransform(imgMask, dist, CV_DIST_L2, 3);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        normalize(dist, dist, 0, 1., NORM_MINMAX);
        //imshow("Distance Transform Image", dist);
        
        // Threshold to obtain the peaks
        // This will be the markers for the foreground objects
        threshold(dist, dist, 0.4, 1., CV_THRESH_BINARY);
        // Dilate a bit the dist image
        Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
        dilate(dist, dist, kernel1);
        //imshow("Peaks", dist);
        
        // Create the CV_8U version of the distance image
        // It is needed for findContours()
        Mat dist_8u;
        dist.convertTo(dist_8u, CV_8U);
        // Find total markers
        vector<vector<Point> > contours;
        findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        // Create the marker image for the watershed algorithm
        Mat markers = Mat::zeros(dist.size(), CV_32SC1);
        // Draw the foreground markers
        for (size_t i = 0; i < contours.size(); i++)
            drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
        // Draw the background marker
        circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
        //imshow("Markers", markers*10000);
        
        // Perform the watershed algorithm
        watershed(imgSharp, markers);
        Mat mark = Mat::zeros(markers.size(), CV_8UC1);
        markers.convertTo(mark, CV_8UC1);
        bitwise_not(mark, mark);
        //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
        // image looks like at that point
        // Generate random colors
        vector<Vec3b> colors;
        for (size_t i = 0; i < contours.size(); i++)
        {
            int b = theRNG().uniform(0, 255);
            int g = theRNG().uniform(0, 255);
            int r = theRNG().uniform(0, 255);
            colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
        }
        // Create the result image
        Mat dst = Mat::zeros(markers.size(), CV_8UC3);
        // Fill labeled objects with random colors
        for (int i = 0; i < markers.rows; i++)
        {
            for (int j = 0; j < markers.cols; j++)
            {
                int index = markers.at<int>(i,j);
                if (index > 0 && index <= static_cast<int>(contours.size()))
                    dst.at<Vec3b>(i,j) = colors[index-1];
                else
                    dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }
        }
        // Visualize the final image
        imshow("Final Result", dst);
        imshow("Result", imgResult );
        
        
        
        // Get the detected lines
        lines = orange::borderLines(dst);
        
        // Try again if no lines found
        if (lines.size() <= 0) continue;
        
        // TRACE: Display the lines on the images
        for(int i = 0; i < lines.size(); i++) {
            Vec4i l = lines[i];
            Point p1 = Point(l[0], l[1]);
            Point p2 = Point(l[2], l[3]);
            
            Scalar colour = Scalar(0,255,0);
            if (i >= 4) colour = Scalar(0,0,255);
            
            line(img, p1, p2, colour, 1);
            line(imgResult, p1, p2, colour, 1);
        }
        
        // TRACE: [Temporary] Only try if > 4 lines
        //if (lines.size() < 4) continue ;
        //lines.resize(4);
        
        // Create the Mat of edge endpoints
        Mat target = edgy::edgeToPointsMat(lines[0]);
        for (int i = 1; i < lines.size(); i++) {
            Mat edgePts = edgy::edgeToPointsMat(lines[i]);
            hconcat(target, edgePts, target);
        }
        vector<Point2f> imgPoints = matToPoints(target);
        
        // Initially choose the basis to be the first edge detected
        vector<int> imgBasis = {0,1};
        
        

        // * * * * * * * * * * * * * *
        //      RECOGNITION
        // * * * * * * * * * * * * * *
        
        auto startRecog = chrono::system_clock::now(); // Start recognition timer

        vector<HashTable> votedTables = hashing::voteForTables(tables, imgPoints, imgBasis);

        // Use least squares to match the tables with the most votes
        int maxVotes = votedTables[0].votes;
        cout << "MAX VOTES = " << maxVotes << endl << endl;
        
        vector<estimate> estList;
        
        for (int i = 0; i < votedTables.size(); i++) {
            HashTable t = votedTables[i];
            if (t.votes < MIN(200, maxVotes)) break;
            
            vector<Mat> orderedPoints = hashing::getOrderedPoints(imgBasis, t, modelPoints, imgPoints);
            
            Mat newModel = orderedPoints[0];
            Mat newTarget = orderedPoints[1];
            
            // Take at most 4 correspondences
            if (newModel.cols > 4) {
                newModel = newModel.colRange(0, 4);
                newTarget = newTarget.rowRange(0, 4);
            }
            
            float xAngle = dA * (0.5 + t.viewAngle[0]);
            float yAngle = (dA * (0.5 + t.viewAngle[1])) - CV_PI/2;
            Vec6f poseInit = {0, 0, defaultZ, xAngle, yAngle, 0};
            estimate est = lsq::poseEstimateLM(poseInit, newModel, newTarget, K);
            
            // Only take projections that converged and are in front of the camera
            if (est.iterations != lsq::MAX_ITERATIONS && est.pose[2] > 0) {
                //TRACE
                cout << "Basis = " << t.basis[0] << "," << t.basis[1] << " | Angle = " << t.viewAngle[0] << "," << t.viewAngle[1] << " | Votes = " << t.votes <<  endl;
                
                est.print();
                estList.push_back(est);
            }
        }
        
        auto endRecog = chrono::system_clock::now();
        
        //TRACE:
        chrono::duration<double> timeRecog = endRecog-startRecog;
        cout << "Recognition time = " << timeRecog.count()*1000.0 << " ms" << endl;
        cout << endl << estList.size() << "/" << votedTables.size() << " successes!" << endl;
        
        if (estList.size() > 0) {
            Mat proj = lsq::projection(estList[0].pose, modelMat, K);
            vector<Point2f> projPts = matToPoints(proj);
            line(imgResult, Point(projPts[0]), Point(projPts[1]), Scalar(255,255,255));
            line(imgResult, Point(projPts[1]), Point(projPts[2]), Scalar(255,255,255));
            line(imgResult, Point(projPts[2]), Point(projPts[3]), Scalar(255,255,255));
            line(imgResult, Point(projPts[3]), Point(projPts[0]), Scalar(255,255,255));
            /*
            line(imgResult, Point(projPts[4]), Point(projPts[5]), Scalar(255,255,255));
            line(imgResult, Point(projPts[5]), Point(projPts[6]), Scalar(255,255,255));
            line(imgResult, Point(projPts[6]), Point(projPts[7]), Scalar(255,255,255));
            line(imgResult, Point(projPts[7]), Point(projPts[4]), Scalar(255,255,255));
            
            line(imgResult, Point(projPts[0]), Point(projPts[4]), Scalar(255,255,255));
            line(imgResult, Point(projPts[1]), Point(projPts[5]), Scalar(255,255,255));
            line(imgResult, Point(projPts[2]), Point(projPts[6]), Scalar(255,255,255));
            line(imgResult, Point(projPts[3]), Point(projPts[7]), Scalar(255,255,255));*/
        }
        
        // TRACE: Display the images
        imshow("imgResult", imgResult);
        imshow("img", img);
        
        // Press 'w' to escape
        if(waitKey(0) == 'w') {cout << "***\n"; break;}
        
    }
    
    
    chrono::duration<double> timeHash = endHash-startHash;
    cout << "Hashing time     = " << timeHash.count()*1000.0 << " ms" << endl;

    return 0;
}




// * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//      Additional Functions
// * * * * * * * * * * * * * * * * * * * * * * * * * * * *

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
