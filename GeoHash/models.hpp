//
//  models.hpp
//  GeoHash
//
//  Created by Daniel Mesham on 16/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef models_hpp
#define models_hpp

#include <opencv2/core/core.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

class Box {
public:
    Box(float width, float height, float depth) {
        createPoints(width, height, depth);
    }
    bool vertexIsVisible(int vertexID, float xAngle, float yAngle);
    vector<bool> visibilityMask(float xAngle, float yAngle);
    vector<Point3f> vertices;
    Mat pointsToMat();
    
    static const vector<vector<int>> edgeBasisList;
    
private:
    static const vector<vector<float>> xAngleLimits;
    static const vector<vector<float>> yAngleLimits;
    void createPoints(float width, float height, float depth);
    
};


/*
 
   z
  /             Rotation directions defined by
 .__ x          the right hand grip rule.
 |
 y
 
   4 .___. 5
    /   /|      Vertex IDs increase in x, y then z directions.
 0 .___.1. 6    i.e. front face then back face
   |   |/
 3 .___. 2
 
 */

#endif /* models_hpp */
