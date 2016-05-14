// Copyright 2016 Dolotov Evgeniy

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <vector>

#include <deep_pyramid.h>
#include "rectangle_transform.h"
#include <fddb_container.h>

#include <string>

using namespace cv;
using namespace std;
using namespace caffe;

static const char argsDefs[] =
        "{ | config           |     | Path to configuration file }";

void printHelp(std::ostream& os) {
    os << "\tUsage: --config=path/to/config.xm" << std::endl;
}

namespace ReturnCode {
enum {
    Success = 0,
    ConfigFileNotSpecified = 1,
    ConfigFileNotFound = 2,
    ImageFileNotFound = 3,
    TrainFileNotFound = 4
};
};

int parseCommandLine(int argc, char *argv[], FileStorage& config) {
    cv::CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");

    if (configFileName.empty() == true) {
        std::cerr << "Configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }

    config.open(configFileName, FileStorage::READ);

    if (config.isOpened() == false) {
        std::cerr << "File '" << configFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ConfigFileNotFound;
    }

    return ReturnCode::Success;
}

#define MARGIN_THRESHOLD 1

void loadFeature(string filename, vector<FeatureMap>& features,string prefix) {
    ifstream file(filename);
    string featureFile;
    while (file >> featureFile) {
        FeatureMap map;
        map.load(prefix+featureFile);
        cout<<prefix+featureFile<<endl;
        if (map.size().area() > 0) {
            features.push_back(map);
        } else {
            cout << featureFile << endl;
            continue;
        }
    }
    cout << "positive:" << features.size() << endl;
}

int main(int argc, char *argv[]) {
    FileStorage config;
    int returnCode = parseCommandLine(argc, argv, config);

    if (returnCode!=ReturnCode::Success) {
        return returnCode;
    }
    // load svm size
    Size filterSize;
    config["filter_size"] >> filterSize;

    string prefix;
    config["prefix"]>>prefix;
    // load positie samples
    string objects_feature_filename;
    config["objects_feature"] >> objects_feature_filename;
    vector<FeatureMap> objectsFeature;
    loadFeature(objects_feature_filename, objectsFeature,prefix);
    cout << "load positive" << endl;
    // load negative samples
    string negative_feature_filename;
    config["negative_feature"] >> negative_feature_filename;
    vector<FeatureMap> negativeFeature;
    loadFeature(negative_feature_filename, negativeFeature,prefix);
    cout << "load negatives" << endl;


    DeepPyramid pyramid(config);

    cout << "train" << endl;
    // train
    FeatureMapSVM svm(filterSize);
    svm.train(objectsFeature, negativeFeature);
    svm.printAccuracy(objectsFeature, negativeFeature);


    int iterationCount;
    config["max_iter"] >> iterationCount;
    FDDBContainer data;
    string train_data_filename;
    string train_data_folder;
    config["train_data"] >> train_data_filename;
    config["train_data_folder"] >> train_data_folder;
    cout<<train_data_filename<<endl <<train_data_folder<<endl;
    data.load(train_data_filename, train_data_folder);
    for (int iter = 0; iter < iterationCount; iter++) {
        string image_path;
        Mat image;
        vector<Rect> objects;
        cout<<"loooool"<<endl;
        data.next(image_path, image, objects);

        vector<FeatureMap> omaps, nmaps;
        pyramid.extractFeatureMap(image, objects, filterSize, omaps, nmaps);
       do
        {
            int newPositive=0;
            for(size_t i=0;i<omaps.size();i++)
            {
                if(newPositive<20)
                {
                    if(svm.predict(omaps[i])==NOT_OBJECT)
                    {
                        objectsFeature.push_back(omaps[i]);
                        newPositive++;
                    }
                }
                else
                {
                    break;
                }
            }
            int newNegative=0;
            for(size_t i=0;i<nmaps.size();i++)
            {
                if(newNegative<40)
                {
                    if(svm.predict(nmaps[i])==OBJECT)
                    {
                        negativeFeature.push_back(nmaps[i]);
                        newNegative++;
                    }
                }
                else
                {
                    break;
                }

            }
            cout << "start train" << endl;
            svm.train(objectsFeature, negativeFeature);
            svm.save("linear_svm"+std::to_string((long long int)iter)+".xml");

            for (unsigned int j = 0; j < objectsFeature.size(); j++) {
                    objectsFeature[j].save(prefix+"positive"+std::to_string((long long unsigned int)j)+".xml");
            }

            for (unsigned int j = 0; j < negativeFeature.size(); j++) {
                    negativeFeature[j].save(prefix+"negative"+std::to_string((long long unsigned int)j)+".xml");
            }
        }
        while(svm.printAccuracy(omaps, nmaps)!=1);
    }

    config.release();

    return ReturnCode::Success;
}
