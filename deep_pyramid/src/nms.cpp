// Copyright 2016 Dolotov Evgeniy

#include "../include/nms.h"
#include <algorithm>
#include <iostream>
#include <vector>

#include <opencv2/objdetect/objdetect.hpp>
#include "../include/rectangle_transform.h"

using namespace std;
using namespace cv;

void NMS::divideIntoClusters(vector<BoundingBox>& objects, const double &box_threshold, vector<BoundingBoxCluster>& clusters) {
    while (!objects.empty()) {
        BoundingBox objectWithMaxConfidence = *max_element(objects.begin(), objects.end());
        BoundingBoxCluster cluster;
        vector<BoundingBox> newObjects;
        for (unsigned int i = 0; i < objects.size(); i++) {
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

void NMS::processBondingBox(vector<BoundingBox> &objects, const double &box_threshold, const double &confidence_threshold) {
    vector<BoundingBox> detectedObjects;
    vector<BoundingBoxCluster> clusters;

    divideIntoClusters(objects, box_threshold, clusters);
    for (vector<BoundingBoxCluster>::iterator cluster = clusters.begin(); cluster != clusters.end(); cluster++) {
        int boundBoxCount = cluster->size();
        cout << "Box in cluster:" << boundBoxCount << endl;
        detectedObjects.push_back(mergeCluster(*cluster, confidence_threshold));
    }
    objects = detectedObjects;
}

BoundingBox NMSmax::mergeCluster(BoundingBoxCluster &cluster, const double &confidence_threshold) {
    return *max_element(cluster.begin(), cluster.end());
}

BoundingBox NMSavg::mergeCluster(BoundingBoxCluster &cluster, const double &confidence_threshold) {
    BoundingBox boundingBoxWithMaxConfidence = *max_element(cluster.begin(), cluster.end());
    double maxConfidenceInCluster = boundingBoxWithMaxConfidence.confidence;

    vector<Rect> rectangleWithMaxConfidence;
    for (vector<BoundingBox>::iterator boundingBox = cluster.begin(); boundingBox != cluster.end(); boundingBox++) {
        if (boundingBox->confidence > confidence_threshold*maxConfidenceInCluster) {
            rectangleWithMaxConfidence.push_back(boundingBox->originalImageBox);
        }
    }

    BoundingBox resultBoundingBox;
    resultBoundingBox.originalImageBox = avg_rect(rectangleWithMaxConfidence);
    resultBoundingBox.confidence = maxConfidenceInCluster;

    return resultBoundingBox;
}

BoundingBox NMSintersect::mergeCluster(BoundingBoxCluster &cluster, const double &confidence_threshold) {
    BoundingBox boundingBoxWithMaxConfidence = *max_element(cluster.begin(), cluster.end());
    double maxConfidenceInCluster = boundingBoxWithMaxConfidence.confidence;

    vector<Rect> rectangleWithMaxConfidence;
    for (vector<BoundingBox>::iterator boundingBox = cluster.begin(); boundingBox != cluster.end(); boundingBox++) {
        if (boundingBox->confidence > confidence_threshold*maxConfidenceInCluster) {
            rectangleWithMaxConfidence.push_back(boundingBox->originalImageBox);
        }
    }

    BoundingBox resultBoundingBox;
    resultBoundingBox.originalImageBox = intersectRectangles(rectangleWithMaxConfidence);
    resultBoundingBox.confidence = maxConfidenceInCluster;

    return resultBoundingBox;
}
