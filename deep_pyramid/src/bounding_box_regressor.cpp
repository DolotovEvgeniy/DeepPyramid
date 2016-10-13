// Copyright 2016 Dolotov Evgeniy

#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "bounding_box_regressor.h"
#include "bounding_box.h"
#include "rectangle_transform.h"

using namespace std;
using namespace cv;

void BoundingBoxRegressor::processBoundingBoxes(vector<BoundingBox>& objects) {
    for (size_t i = 0; i < objects.size(); i++) {
        regressBox(objects[i]);
    }
}

double matToScalar(const Mat& mat) {
    return mat.at<double>(0, 0);
}

void BoundingBoxRegressor::regressBox(BoundingBox& object) {
    Rect rect = object.originalImageBox;

    Point rectCenter = getRectangleCenter(rect);

    Mat feature;
    object.map.reshapeToVector(feature);

    Point newRectCenter;
    newRectCenter.x = rect.width*matToScalar(xWeights*feature)+rectCenter.x;
    newRectCenter.y = rect.width*matToScalar(yWeights*feature)+rectCenter.y;

    int newRectWidth, newRectHeight;
    newRectWidth = rect.width*exp(matToScalar(widthWeights*feature));
    newRectHeight = rect.height*exp(matToScalar(heightWeights*feature));

    Rect newRect = makeRectangle(newRectCenter, newRectWidth, newRectHeight);

    object.originalImageBox = newRect;
}
