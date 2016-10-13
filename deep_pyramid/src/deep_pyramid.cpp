// Copyright 2016 Dolotov Evgeniy

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "deep_pyramid.h"
#include "nms.h"
#include "rectangle_transform.h"

using namespace cv;
using namespace std;

DeepPyramid::DeepPyramid(const FileStorage& config) {
    string model_file;
    string trained_net_file;
    config["net"] >> model_file;
    config["weights"] >> trained_net_file;
    net = new NeuralNetwork(model_file, trained_net_file);

    int levelCountInt;
    config["number_of_levels"] >> levelCountInt;
    levelCount = (size_t)levelCountInt;

    string svm_trained_file;
    config["svm"] >> svm_trained_file;
    Size filterSize;
    config["filter_size"] >> filterSize;
    FeatureMapSVM svm(filterSize);
    svm.load(svm_trained_file);
    rootFilter.push_back(svm);
    config["stride"] >> stride;
}

void DeepPyramid::detect(const Mat& img, vector<BoundingBox>& objects,
                         bool isBoundingBoxRegressor) const {
    CV_Assert(img.channels() == 3);
    vector<Mat> imgPyramid;
    constructImagePyramid(img,imgPyramid);

    vector<FeatureMap> maps;
    constructFeatureMapPyramid(imgPyramid, maps);

    cout << "filter" << endl;
    detect(maps, objects);
    cout << "group rectangle" << endl;
    calculateOriginalRectangle(objects, Size(img.cols, img.rows));
    groupRectangle(objects);
    if (isBoundingBoxRegressor) {
        cout << "boundbox regressor: TODO" << endl;
    } else {
        cout << "bounding box regressor switch off" << endl;
    }
    cout << "Object count:" << objects.size() << endl;
}

void DeepPyramid::extractFeatureMap(const Mat &img, vector<Rect>& objects,
                                    const Size& size,
                                    vector<FeatureMap>& positiveMaps,
                                    vector<FeatureMap>& negativeMaps) {
    Size imgSize(img.cols, img.rows);
    positiveMaps.clear();
    negativeMaps.clear();

    vector<Mat> imgPyramid;
    constructImagePyramid(img,imgPyramid);

    vector<FeatureMap> featureMaps;
    constructFeatureMapPyramid(imgPyramid, featureMaps);

    for (size_t i = 0; i < levelCount; i++) {
        vector<Rect> objectsAtLevel;
        for (size_t obj = 0; obj < objects.size(); obj++) {
            Rect norm5Rect = originalRect2Norm5(objects[obj], i, imgSize);
            objectsAtLevel.push_back(norm5Rect);
        }

        Size mapSize = featureMaps[i].size();
        for (int x = 0; x < mapSize.width-size.width; x+=stride)
            for (int y = 0; y < mapSize.height-size.height; y+=stride) {
                Rect boundingRect(Point(x, y), size);
                for (size_t obj = 0; obj < objects.size(); obj++) {
                    double IOUvalue = IOU(boundingRect, objectsAtLevel[obj]);
                    if (IOUvalue < NEGATIVE_THRESHOLD) {
                        FeatureMap map;
                        featureMaps[i].extractFeatureMap(boundingRect, map);
                        negativeMaps.push_back(map);
                    } else if (IOUvalue > POSITIVE_THRESHOLD ) {
                        FeatureMap map;
                        featureMaps[i].extractFeatureMap(boundingRect, map);
                        positiveMaps.push_back(map);
                    }
                }

            }
    }
}

Size DeepPyramid::embeddedImageSize(const Size& imgSize,
                                    const int& level) const {
    Size networkInputSize = net->inputLayerSize();
    Size newImgSize;
    double scale = 1/imageScale(level);
    double aspectRatio = imgSize.height/(double)imgSize.width;

    if (aspectRatio < 1) {
        newImgSize.width = networkInputSize.width*scale;
        newImgSize.height = newImgSize.width*aspectRatio;
    } else {
        newImgSize.height = networkInputSize.height*scale;
        newImgSize.width = newImgSize.height/aspectRatio;
    }

    return newImgSize;
}

void DeepPyramid::constructImagePyramid(const Mat& img,
                                        vector<Mat>& imgPyramid) const {
    Size imgSize(img.cols, img.rows);
    imgPyramid.clear();
    imgPyramid.reserve(levelCount);

    for (size_t level = 0; level < levelCount; level++) {
        Mat imgAtLevel(net->inputLayerSize(), CV_8UC3, Scalar::all(0));

        Mat resizedImg;
        Size resizedImgSize = embeddedImageSize(imgSize, level);
        resize(img, resizedImg, resizedImgSize);

        resizedImg.copyTo(imgAtLevel(Rect(Point(0, 0), resizedImgSize)));

        imgPyramid[level] = imgAtLevel;
    }
}

void DeepPyramid::constructFeatureMapPyramid(const vector<Mat>& imgPyramid,
                                             vector<FeatureMap>& maps) const {
    maps.clear();
    maps.reserve(levelCount);

    for (size_t i = 0; i < levelCount; i++) {
        FeatureMap map;
        net->processImage(imgPyramid[i], map);
        map.normalize();
        maps[i] = map;
    }
}


void DeepPyramid::processFeatureMap(const int& filterIdx, const FeatureMap& map,
                                  vector<BoundingBox>& detectedObjects) const {
    const FeatureMapSVM& filter = rootFilter[filterIdx];
    Size filterSize = filter.getMapSize();


    Rect boundingRectangle(Point(0, 0), filterSize);

    Size mapSize = map.size();
    int xEnd = mapSize.width-filterSize.width;
    int yEnd = mapSize.height-filterSize.height;

    for (int x = 0; x < xEnd; x+=stride) {
        for (int y = 0; y < yEnd; y+=stride) {
            boundingRectangle.x = x;
            boundingRectangle.y = y;

            FeatureMap rectMap;
            map.extractFeatureMap(boundingRectangle, rectMap);

            if (filter.predictObjectType(rectMap) == OBJECT) {
                BoundingBox box;
                box.norm5Box = boundingRectangle;
                box.confidence = std::fabs(filter.predictConfidence(rectMap));
                box.map = rectMap;
                detectedObjects.push_back(box);
            }
        }
    }
}

void DeepPyramid::detect(const vector<FeatureMap>& maps,
                         vector<BoundingBox>& detectedObjects) const {
    detectedObjects.clear();

    for (size_t i = 0; i < rootFilter.size(); i++)
        for (size_t j = 0; j < levelCount; j++) {
            vector<BoundingBox> detectedObjectsAtLevel;
            processFeatureMap(i, maps[j], detectedObjectsAtLevel);
            for (size_t k = 0; k < detectedObjectsAtLevel.size(); k++) {
                detectedObjectsAtLevel[k].level = j;
                detectedObjects.push_back(detectedObjectsAtLevel[k]);
            }
        }
}

double DeepPyramid::imageScale(const int& level) const{
    return pow(2, (levelCount-1-level)/2.0);
}

Rect DeepPyramid::norm5Rect2Original(const Rect& norm5Rect, const int& level,
                                     const Size& imgSize) const {
    double longSide = std::max(imgSize.height, imgSize.width);
    Size networkOutputSize = net->outputLayerSize();
    double scale = (imageScale(level)*longSide)/networkOutputSize.width;
    return scaleRect(norm5Rect, scale);
}

Rect DeepPyramid::originalRect2Norm5(const Rect& originalRect,
                                     const int& level,
                                     const Size& imgSize) const {
    double longSide = std::max(imgSize.height, imgSize.width);

    Size networkOutputSize = net->outputLayerSize();
    double scale = networkOutputSize.width/(imageScale(level)*longSide);

    Rect rect = scaleRect(originalRect, scale);
    if (rect.x < 0)
        rect.x = 0;
    if (rect.y < 0)
        rect.y = 0;
    if (rect.x+rect.width > networkOutputSize.width) {
        rect.width = networkOutputSize.width-rect.x-1;
    }
    if (rect.y+rect.height > networkOutputSize.height) {
        rect.height = networkOutputSize.height-rect.y-1;
    }
    return rect;
}

int DeepPyramid::chooseLevel(const Size& filterSize, const Rect& boundBox,
                             const Size& imgSize) const {
    vector<double> f;
    f.reserve(levelCount);
    for (size_t i = 0; i < levelCount; i++) {
        Rect r = originalRect2Norm5(boundBox, i, imgSize);

        f[i] = fabs(filterSize.width-r.width)+abs(r.height-filterSize.height);
    }
    int bestLevel = distance(f.begin(), min_element(f.begin(), f.end()));

    return bestLevel;
}

void DeepPyramid::calculateOriginalRectangle(vector<BoundingBox>& detectedObjects,
                                             const Size& imgSize) const {
    for (size_t i = 0; i < detectedObjects.size(); i++) {
        Rect originalRect = norm5Rect2Original(detectedObjects[i].norm5Box,
                                               detectedObjects[i].level, imgSize);
        detectedObjects[i].originalImageBox = originalRect;
    }
}

void DeepPyramid::groupRectangle(vector<BoundingBox>& detectedObjects) const {
    NMSweightedAvg nms;
    nms.processBondingBox(detectedObjects, BOX_THRESHOLD, CONFIDENCE_THRESHOLD);
}

void DeepPyramid::detect(const Mat& img, vector<Rect>& detectedObjects,
                         vector<float>& confidence,
                         bool isBoundingBoxRegressor) const {
    CV_Assert(img.channels() == 3);

    vector<Mat> imgPyramid;
    constructImagePyramid(img,imgPyramid);

    vector<FeatureMap> maps;
    constructFeatureMapPyramid(imgPyramid, maps);

    cout << img.cols << "," << img.rows << "," << rootFilter.size() << endl;
    vector<BoundingBox> objects;
    detect(maps, objects);
    cout << "group rectangle" << endl;
    calculateOriginalRectangle(objects, Size(img.cols, img.rows));
    groupRectangle(objects);
    if (isBoundingBoxRegressor) {
        cout << "boundbox regressor: TODO" << endl;
    } else {
        cout << "bounding box regressor switch off" << endl;
    }
    cout << "Object count:" << objects.size() << endl;

    detectedObjects.clear();
    confidence.clear();
    for (size_t i = 0; i < objects.size(); i++) {
        detectedObjects.push_back(objects[i].originalImageBox);
        confidence.push_back(objects[i].confidence);
    }
}
