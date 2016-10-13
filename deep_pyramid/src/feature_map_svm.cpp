// Copyright 2016 Dolotov Evgeniy

#include <string>
#include <vector>
#include <iostream>

#include "feature_map_svm.h"

using namespace std;
using namespace cv;

void FeatureMapSVM::load(const string& filename) {
    svm->load(filename.c_str());
}

void FeatureMapSVM::save(const string& filename) const{
    svm->save(filename.c_str());
}

ObjectType FeatureMapSVM::predictObjectType(const FeatureMap& sample) const {
    Mat feature;
    sample.reshapeToVector(feature);

    double predictValue =  svm->predict(feature);
    if(predictValue == 1) {
        return OBJECT;
    } else {
        return NOT_OBJECT;
    }
}

double FeatureMapSVM::predictConfidence(const FeatureMap& sample) const {
    Mat feature;
    sample.reshapeToVector(feature);

    return svm->predict(feature, true);
}

void FeatureMapSVM::train(const vector<FeatureMap>& positive,
                          const vector<FeatureMap>& negative) {
    cout<<"Positives count: "<<positive.size()<<endl;
    cout<<"Negatives count: "<<negative.size()<<endl;
    Mat features;
    Mat labels;

    for (size_t i = 0; i < positive.size(); i++) {
        Mat feature;

        positive[i].reshapeToVector(feature);
        features.push_back(feature);
        labels.push_back(OBJECT);
    }

    for (size_t i = 0; i < negative.size(); i++) {
        Mat feature;

        negative[i].reshapeToVector(feature);
        features.push_back(feature);
        labels.push_back(NOT_OBJECT);
    }

    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

    svm->train_auto(features, labels, Mat(), Mat(), params);
}

FeatureMapSVM::~FeatureMapSVM() {
    delete svm;
}

float FeatureMapSVM::printAccuracy(const vector<FeatureMap>& positive,
                                   const vector<FeatureMap>& negative) const{
    int objectsCount = positive.size();
    int negativeCount = negative.size();

    int trueNegative = 0;
    int truePositive = 0;

    for (size_t i = 0; i < positive.size(); i++) {
        if (predictObjectType(positive[i]) == OBJECT) {
            truePositive++;
        }
    }
    for (size_t i = 0; i < negative.size(); i++) {
        if (predictObjectType(negative[i]) == NOT_OBJECT) {
            trueNegative++;
        }
    }
    cout << "Objects classification accuracy:" 
         << truePositive/(double)objectsCount
         << endl;
    
    cout << "Negative classification accuracy:" 
         << trueNegative/(double)negativeCount 
         << endl;
    
    cout << "Common classification accuracy:"
         << (truePositive+trueNegative)/(double)(objectsCount+negativeCount)
         << endl;

    cout << "Count of features:" << objectsCount+negativeCount << endl;

    return trueNegative/(double)negativeCount;
}

Size FeatureMapSVM::getMapSize() const{
    return mapSize;
}
FeatureMapSVM::FeatureMapSVM(Size size) {
    mapSize = size;
    svm = new CvSVM();
}
