//
//  orange.hpp
//  LiveTracker
//
//  Created by Daniel Mesham on 01/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef orange_hpp
#define orange_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

class orange {
public:
    static vector<Point> getOrangeCorners(Mat img);
    static vector<Vec4i> borderLines(Mat img);
    static vector<Point> linesToCorners(vector<Vec4i> lines);
    static Mat segmentByColour(Mat img, Scalar colour);
    
private:
    static vector<vector<Point>> sortSides(vector<vector<Point>> sides);
    static Point intersection(vector<Point> a, vector<Point> b);
};

#endif /* orange_hpp */
