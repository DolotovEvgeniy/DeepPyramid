// Copyright 2016 Dolotov Evgeniy

#include <deep_pyramid.h>
#include <nms.h>
#include <rectangle_transform.h>

#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <string>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

#define TIMER_START(name) int64 t_##name = getTickCount()
#define TIMER_END(name) printf(#name ":\t%6.2fms\n", \
    1000.f * ((getTickCount() - t_##name) / getTickFrequency()))

Rect DeepPyramid::norm5Rect2Original(const Rect& norm5Rect, int level, const Size& imgSize) const {
    double longSide = std::max(imgSize.height, imgSize.width);
    Size networkOutputSize = net->outputLayerSize();
    double scale = (pow(2, (levelCount-1-level)/2.0)*longSide)/networkOutputSize.width;
    return scaleRect(norm5Rect, scale);
}

Size DeepPyramid::embeddedImageSize(const Size& imgSize, const int& i) const {
    Size networkInputSize = net->inputLayerSize();
    Size newImgSize;
    double scale = 1/pow(2, (levelCount-1-i)/2.0);
    double aspectRatio = imgSize.height/(double)imgSize.width;
    if (imgSize.height <= imgSize.width) {
        newImgSize.width = networkInputSize.width*scale;
        newImgSize.height = newImgSize.width*aspectRatio;
    } else {
        newImgSize.height = networkInputSize.height*scale;
        newImgSize.width = newImgSize.height/aspectRatio;
    }

    return newImgSize;
}

void DeepPyramid::constructImagePyramid(const Mat& img, vector<Mat>& imgPyramid) const {
    Size imgSize(img.cols, img.rows);

    TIMER_START(Create_image_pyramid);
    for (int level = 0; level < levelCount; level++) {
        int size = 1713;
        size *= 1/pow(2, (levelCount-1-level)/2.0);
        Mat imgAtLevel(Size(size, size), CV_8UC3, Scalar::all(0));

        Mat resizedImg;
        Size resizedImgSize = embeddedImageSize(imgSize, level);
        resize(img, resizedImg, resizedImgSize);
        resizedImg.copyTo(imgAtLevel(Rect(Point(0, 0), resizedImgSize)));
        imgPyramid.push_back(imgAtLevel);
    }
    TIMER_END(Create_image_pyramid);

    cout << "Status: Success!" << endl;
}

void DeepPyramid::constructFeatureMapPyramid(const Mat& img, vector<FeatureMap>& maps) const {
    vector<Mat> imgPyramid;
    constructImagePyramid(img, imgPyramid);
    for (int i = 0; i < levelCount; i++) {
        TIMER_START(NeuralNetwork_Compute);
        FeatureMap map;
        net->processImage(imgPyramid[i], map);

        map.normalize();
        cout << "jjjj" << map.size() <<endl;
        maps.push_back(map);
        TIMER_END(NeuralNetwork_Compute);
    }
}
//
//
void DeepPyramid::detect(const vector<FeatureMap>& maps, vector<BoundingBox>& detectedObjects) const {
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
//
//

// Rectangle
//

void DeepPyramid::calculateOriginalRectangle(vector<BoundingBox>& detectedObjects, const Size& imgSize) const {
    for (size_t i = 0; i < detectedObjects.size(); i++) {
        Rect originalRect = norm5Rect2Original(detectedObjects[i].norm5Box, detectedObjects[i].level, imgSize);
        detectedObjects[i].originalImageBox = originalRect;
    }
}

void DeepPyramid::groupRectangle(vector<BoundingBox>& detectedObjects) const {
    NMSweightedAvg nms;
    nms.processBondingBox(detectedObjects, 0.2, 0.7);
}

void DeepPyramid::detect(const Mat& img, vector<Rect>& detectedObjects, vector<float>& confidence, bool isBoundingBoxRegressor) const {
    CV_Assert(img.channels() == 3);
    vector<FeatureMap> maps;
    constructFeatureMapPyramid(img, maps);
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
    for (size_t i = 0; i < objects.size(); i++) {
        detectedObjects.push_back(objects[i].originalImageBox);
        confidence.push_back(objects[i].confidence);
    }
}

DeepPyramid::DeepPyramid(string model_file, string trained_net_file,
                         vector<string> svm_file, vector<Size> svmSize,
                         int _levelCount, int _stride) {
    net = new NeuralNetwork(model_file, trained_net_file);
    levelCount = _levelCount;
    stride = _stride;
    for (size_t i = 0; i < svm_file.size(); i++) {
        FeatureMapSVM* svm = new FeatureMapSVM(svmSize[i]);
        svm->load(svm_file[i]);
        rootFilter.push_back(svm);
    }
}

DeepPyramid::DeepPyramid(FileStorage config) {
    string model_file;
    string trained_net_file;
    config["net"] >> model_file;
    config["weights"] >> trained_net_file;
    net = new NeuralNetwork(model_file, trained_net_file);

    config["number_of_levels"] >> levelCount;

    string svm_trained_file;
    config["svm"] >> svm_trained_file;
    Size filterSize;
    config["filter_size"] >> filterSize;
    FeatureMapSVM* svm = new FeatureMapSVM(filterSize);
    svm->load(svm_trained_file);
    rootFilter.push_back(svm);
    config["stride"] >> stride;
}

void DeepPyramid::detect(const Mat &img, vector<BoundingBox> &objects, bool isBoundingBoxRegressor) const {
    CV_Assert(img.channels() == 3);
    vector<FeatureMap> maps;
    cout << "here!" << endl;
    constructFeatureMapPyramid(img, maps);
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

DeepPyramid::~DeepPyramid() {
}

Rect DeepPyramid::originalRect2Norm5(const Rect& originalRect, int level, const Size& imgSize) const {
    double longSide = std::max(imgSize.height, imgSize.width);

    Size networkOutputSize = net->outputLayerSize();
    double scale = networkOutputSize.width/(pow(2, (levelCount-1-level)/2.0)*longSide);

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

int DeepPyramid::chooseLevel(const Size& filterSize, const Rect& boundBox, const Size& imgSize) const {
    vector<double> f;
    for (int i = 0; i < levelCount; i++) {
        Rect r = originalRect2Norm5(boundBox, i, imgSize);

        f.push_back(abs(filterSize.width-r.width)+abs(r.height-filterSize.height));
    }
    int bestLevel = distance(f.begin(), min_element(f.begin(), f.end()));

    return bestLevel;
}

void DeepPyramid::changeRootFilter(FeatureMapSVM* svm) {
    rootFilter.clear();
    rootFilter.push_back(svm);
}

void DeepPyramid::extractFeatureMap(const Mat &img, vector<Rect> &objects, Size size,
                                              vector<FeatureMap> &omaps, vector<FeatureMap>& nmaps) {
    Size imgSize(img.cols, img.rows);

    vector<FeatureMap> featureMaps;
    constructFeatureMapPyramid(img, featureMaps);

    for (int i = 0; i < levelCount; i++) {
        vector<Rect> objectsAtLevel;
        for (size_t obj = 0; obj < objects.size(); obj++) {
            objectsAtLevel.push_back(originalRect2Norm5(objects[obj], i, imgSize));
        }

        Size mapSize = featureMaps[i].size();
        for (int w = 0; w < mapSize.width-size.width; w+=stride)
            for (int h = 0; h < mapSize.height-size.height; h+=stride) {
                bool isNegative = true;
                bool isPositive = false;
                for (size_t obj = 0; obj < objects.size(); obj++) {
                    if (IOU(Rect(Point(w, h), size), objectsAtLevel[obj]) > 0.3)
                        isNegative = false;
                    if (IOU(Rect(Point(w, h), size), objectsAtLevel[obj]) > 0.8)
                        isPositive = true;
                }
                FeatureMap map;
                if (isNegative) {
                    featureMaps[i].extractFeatureMap(Rect(Point(w, h), size), map);
                    nmaps.push_back(map);
                }
                if (isPositive) {
                    featureMaps[i].extractFeatureMap(Rect(Point(w, h), size), map);
                    omaps.push_back(map);
                }

            }
    }
}

void DeepPyramid::processFeatureMap(int filterIdx, const FeatureMap &map, vector<BoundingBox> &detectedObjects) const {
    Size mapSize = map.size();
    Size filterSize = rootFilter[filterIdx]->getMapSize();
    cout << "size: "<<map.size()<<endl;
    for (int width = 0; width < mapSize.width-filterSize.width; width+=stride) {
        for (int height = 0; height < mapSize.height-filterSize.height; height+=stride) {
            FeatureMap extractedMap;
            map.extractFeatureMap(Rect(Point(width, height), filterSize), extractedMap);
            if (rootFilter[filterIdx]->predict(extractedMap) == OBJECT) {
                BoundingBox box;
                box.norm5Box = Rect(Point(width, height), filterSize);
                box.confidence = std::fabs(rootFilter[filterIdx]->predict(extractedMap, true));
                box.map = extractedMap;
                detectedObjects.push_back(box);
            }
        }
    }
}
