// Copyright 2016 Dolotov Evgeniy

#include <algorithm>
#include <iostream>
#include <vector>

#include <opencv2/objdetect/objdetect.hpp>

#include "nms.h"
#include "rectangle_transform.h"

using namespace std;
using namespace cv;

void NMS::divideIntoClusters(vector<BoundingBox>& objects, const double& box_threshold, vector<BoundingBoxCluster>& clusters) {
    while (!objects.empty()) {
        BoundingBox objectWithMaxConfidence = *max_element(objects.begin(), objects.end());
        BoundingBoxCluster cluster;
        vector<BoundingBox> newObjects;
        for (size_t i = 0; i < objects.size(); i++) {
            if (IOU(objectWithMaxConfidence.originalImageBox, objects[i].originalImageBox) <= box_threshold) {
                newObjects.push_back(objects[i]);
            } else {
                cluster.push_back(objects[i]);
            }
        }
        clusters.push_back(cluster);
        objects = newObjects;
    }
}

void NMS::processBondingBox(vector<BoundingBox>& objects, const double& box_threshold, const double& confidence_threshold) {
    vector<BoundingBox> detectedObjects;
    vector<BoundingBoxCluster> clusters;

    divideIntoClusters(objects, box_threshold, clusters);
    for (vector<BoundingBoxCluster>::iterator cluster = clusters.begin(); cluster != clusters.end(); cluster++) {
        int boundBoxCount = cluster->size();
        cout << "Box in cluster:" << boundBoxCount << endl;
        BoundingBox box = mergeCluster(*cluster, confidence_threshold);
        detectedObjects.push_back(mergeCluster(*cluster, confidence_threshold));
    }
    objects = detectedObjects;
}

BoundingBox NMS::mergeCluster(BoundingBoxCluster& cluster, const double& confidence_threshold) {
    BoundingBox boundingBoxWithMaxConfidence = *max_element(cluster.begin(), cluster.end());
    double maxConfidenceInCluster = boundingBoxWithMaxConfidence.confidence;

    vector<BoundingBox> rectangleWithMaxConfidence;
    for (vector<BoundingBox>::iterator boundingBox = cluster.begin(); boundingBox != cluster.end(); boundingBox++) {
        if (boundingBox->confidence > confidence_threshold*maxConfidenceInCluster) {
            rectangleWithMaxConfidence.push_back(*boundingBox);
        }
    }

    BoundingBox resultBoundingBox;
    resultBoundingBox.originalImageBox = groupBoundingBox(rectangleWithMaxConfidence);
    resultBoundingBox.confidence = maxConfidenceInCluster;
    cout << "Confidence" << maxConfidenceInCluster << endl;
    return resultBoundingBox;
}

Rect NMSweightedAvg::groupBoundingBox(std::vector<BoundingBox>& objects) {
    return weightedAvg_rect(objects);
}
