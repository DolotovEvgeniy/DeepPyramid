#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "bounding_box_regressor.h"
#include "bounding_box.h"

using namespace std;
using namespace cv;

void BoundingBoxRegressor::regress(vector<BoundingBox> &objects, const vector<Mat> &features)
{
    for(unsigned int i=0;i<objects.size();i++)
    {
        regressBox(objects[i], features[i]);
    }
}

void BoundingBoxRegressor::regressBox(BoundingBox& object, const Mat &feature)
{
    Rect detectedRect=object.originalImageBox;

    Point rectCenter;

    rectCenter.x=detectedRect.x+detectedRect.width/2.0;
    rectCenter.y=detectedRect.y+detectedRect.height/2.0;

    int rectWidth=detectedRect.width;
    int rectHeight=detectedRect.height;

    Mat dX,dY,dW,dH;

    Point newRectCenter;

    dX=xWeights*feature;
    newRectCenter.x=rectWidth*dX.at<double>(0, 0)+rectCenter.x;

    dY=yWeights*feature;
    newRectCenter.y=rectWidth*dY.at<double>(0, 0)+rectCenter.y;

    int newRectWidth, newRectHeight;

    dW=widthWeights*feature;
    newRectWidth=rectWidth*exp(dW.at<double>(0, 0));

    dH=heightWeights*feature;
    newRectHeight=rectHeight*exp(dH.at<double>(0, 0));

    Rect newRect;
    newRect.x=newRectCenter.x-newRectWidth/2.0;
    newRect.y=newRectCenter.y-newRectHeight/2.0;
    newRect.width=newRectWidth;
    newRect.height=newRectHeight;

    object.originalImageBox=newRect;
}
