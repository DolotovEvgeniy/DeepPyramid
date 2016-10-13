// Copyright 2016 Dolotov Evgeniy

#include <string>
#include <vector>

#include "feature_map.h"

using namespace std;
using namespace cv;

void FeatureMap::addLayer(Mat layer) {
    map.push_back(layer);
}

Mat uniteMats(std::vector<Mat> m) {
    Mat unite(1, 0, CV_32FC1);
    for (size_t i = 0; i < m.size(); i++) {
        unite.push_back(m[i].reshape(1, 1));
    }
    return unite;
}

void calculateMeanAndDeviationValue(vector<Mat> level, float& meanValue,
                                    float& deviationValue) {
    Mat unite = uniteMats(level);
    Mat mean, deviation;
    // корень или квадрат?
    meanStdDev(unite, mean, deviation);
    meanValue = mean.at<double>(0, 0);
    deviationValue = deviation.at<double>(0, 0);
}

void FeatureMap::normalize() {
    float mean, deviation;
    calculateMeanAndDeviationValue(map, mean, deviation);
    for (size_t layer = 0; layer < map.size(); layer++) {
        map[layer] = (map[layer]-mean)/deviation;
    }
}

void FeatureMap::extractFeatureMap(const Rect& rect,
                                   FeatureMap& extractedMap) const {
    for (size_t layer = 0; layer < map.size(); layer++) {
        extractedMap.addLayer(map[layer](rect));
    }
}

void FeatureMap::resize(const Size& size) {
    for (size_t layer = 0; layer < map.size(); layer++) {
        cv::resize(map[layer], map[layer], size);
    }
}

Size FeatureMap::size() const {
    return Size(map[0].cols, map[0].rows);
}

int FeatureMap::area() const {
    return map[0].cols * map[0].rows;
}

void FeatureMap::reshapeToVector(Mat& feature) const {
    int cols = map[0].cols;
    int rows = map[0].rows;

    feature = Mat(1, rows*cols*map.size(), CV_32FC1);
    for (size_t layer = 0; layer < map.size(); layer++) {
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                int indx = x+y*cols+layer*cols*rows;
                feature.at<float>(0, indx)=map[layer].at<float>(y, x);
            }
        }
    }
}

bool FeatureMap::load(string file_name) {
    FileStorage file(file_name, FileStorage::READ);

    if (file.isOpened() == false) {
        return false;
    }

    int channels;
    file["channels"] >> channels;

    for (int i = 0; i < channels; i++) {
        Mat channel;
        file["channel_"+std::to_string((long long int)i)] >> channel;
        map.push_back(channel);
    }
    file.release();
    return true;
}

bool FeatureMap::save(string file_name) {
    FileStorage file(file_name, FileStorage::WRITE);

    if (file.isOpened() == false) {
        return false;
    }

    file << "channels" << (int)map.size();
    for (size_t i = 0; i < map.size(); i++) {
        file << "channel_"+std::to_string((long long int)i) << map[i];
    }
    file.release();
    return true;
}
