// Copyright 2016 Dolotov Evgeniy

#ifndef SVM_H
#define SVM_H

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>

#include "feature_map.h"

enum ObjectType {
    OBJECT,
    NOT_OBJECT
};

class FeatureMapSVM
{
private:
    CvSVM* svm;
    cv::Size mapSize;

public:
    FeatureMapSVM(cv::Size size);
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    ObjectType predictObjectType(const FeatureMap& sample) const;
    double predictConfidence(const FeatureMap& sample) const;
    void train(const std::vector<FeatureMap>& positive, const std::vector<FeatureMap>& negative);
    float printAccuracy(const std::vector<FeatureMap>& positive, const std::vector<FeatureMap>& negative) const;
    ~FeatureMapSVM();
    cv::Size  getMapSize() const;
};
#endif
