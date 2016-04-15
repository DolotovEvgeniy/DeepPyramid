#ifndef SVM_H
#define SVM_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <vector>

#include <feature_map.h>

#include <caffe/caffe.hpp>
#include <caffe/common.hpp>

#define OBJECT 1
#define NOT_OBJECT -1

class FeatureMapSVM
{
public:
    FeatureMapSVM(cv::Size size);
    void save(const std::string& filename);
    void load(const std::string& filename);
    float predict(const FeatureMap& sample, bool returnDFVal=false) const;
    void train(const std::vector<FeatureMap>& positive, const std::vector<FeatureMap>& negative);
    void printAccuracy(const std::vector<FeatureMap> &positive, const std::vector<FeatureMap>& negative);
    ~FeatureMapSVM();
    cv::Size  getMapSize();
private:
    CvSVM* svm;
    cv::Size mapSize;
};
#endif
