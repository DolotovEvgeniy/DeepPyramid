// Copyright 2016 Dolotov Evgeniy

#include "../include/detect_result_container.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::ofstream;
using std::vector;
using cv::Mat;
using cv::Rect;

void DetectResultContainer::add(string image_path, vector<Rect> objects, vector<float> confidence) {
    imagesPath.push_back(image_path);

    vector<Rect> addedObjects;
    vector<float> addedObjectsConfidence;
    for (unsigned int i = 0; i < objects.size(); i++) {
        addedObjects.push_back(objects[i]);
        addedObjectsConfidence.push_back(confidence[i]);
    }

    objectsList.push_back(addedObjects);
    confidenceList.push_back(addedObjectsConfidence);
}

int DetectResultContainer::detectedObjectsCount() {
    int count = 0;
    for (unsigned int i = 0; i < objectsList.size(); i++) {
        count+=objectsList[i].size();
    }

    return count;
}

int DetectResultContainer::size() {
    return imagesPath.size();
}

void DetectResultContainer::save(string file_name) {
    ofstream file(file_name);

    if (file.is_open() == false) {
        std::cerr << "Output file '" << file_name
                  << "' not created. Exiting" << std::endl;
        return;
    }
    for (int i = 0; i < size(); i++) {
        file << imagesPath[i] << endl;
        file << objectsList[i].size() << endl;
        for (unsigned int j = 0; j < objectsList[i].size(); j++) {
            file << objectsList[i][j].x << " ";
            file << objectsList[i][j].y << " ";
            file << objectsList[i][j].width << " ";
            file << objectsList[i][j].height << " ";
            file << confidenceList[i][j] << endl;
        }
    }
}
