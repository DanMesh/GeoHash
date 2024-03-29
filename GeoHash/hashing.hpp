//
//  hashing.hpp
//  GeoHash
//
//  Created by Daniel Mesham on 15/08/2018.
//  Copyright © 2018 Daniel Mesham. All rights reserved.
//

#ifndef hashing_hpp
#define hashing_hpp

#include "geo_hash.h"

#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

class HashTable {
public:
    HashTable(geo_hash table_in, vector<int> basis_in, vector<int> angle_in = {}, int model_in = -1) : table(table_in), basis(basis_in), viewAngle(angle_in), model(model_in) {};
    
    geo_hash table;
    vector<int> basis;
    vector<int> viewAngle;
    int model;
    int votes = 0;
    
    bool operator > (const HashTable& ht) const
    {
        return (votes > ht.votes);
    }
};

class hashing {
public:
    static HashTable createTable(vector<int> basisID, vector<Point2f> modelPoints, vector<bool> visible, float binWidth, vector<int> angle_in);
    static vector<HashTable> voteForTables(vector<HashTable> tables, vector<Point2f> imgPoints, vector<int> imgBasis);
    static vector<HashTable> clearVotes(vector<HashTable> tables);
    static vector<Mat> getOrderedPoints(vector<int> imgBasis, HashTable ht, vector<Point3f> modelPoints, vector<Point2f> imgPoints);
    static Point2f basisCoords(vector<Point2f> basis, Point2f p);
    static Mat pointsToMat2D(vector<Point2f> points);
    static Mat pointsToMat3D(vector<Point3f> points);
    static Mat pointsToMat3D_Homog(vector<Point3f> modelPoints);
};

#endif /* hashing_hpp */
