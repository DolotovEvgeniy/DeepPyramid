#ifndef DEEP_PYRAMID_H
#define DEEP_PYRAMID_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <math.h>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <time.h>
#include <opencv2/objdetect/objdetect.hpp>



enum ClassificationType {FACE, NOT_FACE};

class FaceBox
{
public:
    ClassificationType type;
    double confidence;
    int level;
    cv::Rect norm5Box;
    cv::Rect pyramidImageBox;
    cv::Rect originalImageBox;
};

class DeepPyramid
{
public:
    cv::Mat originalImg;
    cv::Mat originalImgWithFace;
    int num_levels;
    std::vector<cv::Mat> imagePyramid;
    std::vector< std::vector<cv::Mat> > max5;
    std::vector< std::vector<cv::Mat> > norm5;
    std::vector<float> meanValue;
    std::vector<float> deviationValue;
    std::vector<cv::Size> rootFilterSize;
    std::vector<CvSVM*> rootFilterSVM;
    std::vector<FaceBox> allFaces;
    std::vector<cv::Rect> detectedFaces;
    void drawFace();
    caffe::shared_ptr<caffe::Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;

    void detect(cv::Mat img);
    DeepPyramid(int num_levels, const std::string& model_file,
                const std::string& trained_file);
    void showImagePyramid();
    void setImg(cv::Mat img);
    void createMax5PyramidTest();
    void showNorm5Pyramid();
    void addRootFilter(cv::Size filterSize, CvSVM* classifier);
public:
    //Image Pyramid
    cv::Size calculateLevelPyramidImageSize(int i);
    cv::Mat createLevelPyramidImage(int i);
    void createImagePyramid();

    //NeuralNet
    cv::Mat convertToFloat(cv::Mat img);
    void fillNeuralNetInput(int i);
    void calculateNet();
    void calculateNetAtLevel(int i);
    std::vector<cv::Mat> wrapNetOutputLayer();

    //Max5
    std::vector<cv::Mat> createLevelPyramidMax5(int i);
    void createMax5Pyramid();

    //Norm5
    std::vector<cv::Mat> createLevelPyramidNorm5(int i);
    void createNorm5Pyramid();

    //Root-Filter sliding window
    cv::Mat getFeatureVector(int levelIndx, cv::Point position, cv::Size size);
    void rootFilterAtLevel(int rootFilterIndx, int levelIndx, int stride);
    void rootFilterConvolution();

    //Rectangle
    void calculateImagePyramidRectangle();
    void calculateOriginalRectangle();
    void groupOriginalRectangle();

};

#endif // DEEP_PYRAMID_H
