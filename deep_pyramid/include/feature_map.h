#ifndef FEATURE_MAP_H
#define FEATURE_MAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

class FeatureMap
{
public:
    void addLayer(cv::Mat layer);
    void normalize();
    void extractFeatureMap(const cv::Rect& rect, FeatureMap& map) const;
    void resize(const cv::Size& size);
    cv::Size size() const;
    void reshapeToVector(cv::Mat&  feature) const;
    void show() const;
private:
    std::vector<cv::Mat> map;
};

#endif
