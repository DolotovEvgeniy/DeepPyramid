// Copyright 2016 Dolotov Evgeniy

#ifndef FEATURE_MAP_H
#define FEATURE_MAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <string>

class FeatureMap
{
public:
    void addLayer(cv::Mat layer);
    void normalize();
    void extractFeatureMap(const cv::Rect& rect, FeatureMap& map) const;
    void resize(const cv::Size& size);
    cv::Size size() const;
    void reshapeToVector(cv::Mat&  feature) const;
    bool save(std::string file_name);
    bool load(std::string file_name);
private:
    std::vector<cv::Mat> map;
};

#endif
