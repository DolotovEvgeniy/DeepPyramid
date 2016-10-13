// Copyright 2016 Dolotov Evgeniy

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>

#include "feature_map.h"

class NeuralNetwork
{
private:
    caffe::shared_ptr<caffe::Net<float> > net;

public:
    NeuralNetwork() {}
    NeuralNetwork(std::string configFile, std::string trainedModel);
    void processImage(const cv::Mat& img, FeatureMap& map);
    cv::Size inputLayerSize();
    cv::Size outputLayerSize();

private:
    void fillNeuralNetInput(const cv::Mat& img);
    void getNeuralNetOutput(FeatureMap& map);
    void calculate();
};

#endif
