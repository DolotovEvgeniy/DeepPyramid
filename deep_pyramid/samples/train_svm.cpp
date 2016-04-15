#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>

#include <deep_pyramid.h>
#include "rectangle_transform.h"
#include <fddb_container.h>

#include <string>

using namespace cv;
using namespace std;
using namespace caffe;

static const char argsDefs[] =
        "{ | config           |     | Path to configuration file }";

void printHelp(std::ostream& os)
{
    os << "\tUsage: --config=path/to/config.xm" << std::endl;
}

namespace ReturnCode
{
enum
{
    Success = 0,
    ConfigFileNotSpecified = 1,
    ConfigFileNotFound = 2,
    ImageFileNotFound = 3,
    TrainFileNotFound = 4
};
};

int parseCommandLine(int argc, char *argv[], FileStorage& config)
{
    cv::CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");

    if (configFileName.empty() == true)
    {
        std::cerr << "Configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }

    config.open(configFileName, FileStorage::READ);

    if(config.isOpened()==false)
    {
        std::cerr << "File '" << configFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ConfigFileNotFound;
    }

    return ReturnCode::Success;
}

#define MARGIN_THRESHOLD 1

void loadFeature(string filename, int type, Size size, vector<FeatureMap>& features, vector<int>& labels)
{
    ifstream file(filename);
    string featureFile;
    while(file>>featureFile)
    {
        FeatureMap map;
        map.load(featureFile);
        map.resize(size);

        features.push_back(map);
        labels.push_back(type);
    }
}

void calculateAccuracy(CvSVM* svm, Mat& features, Mat& labels)
{
    int objectsCount=0;
    int negativeCount=0;

    int trueNegative=0;
    int truePositive=0;

    for(int i=0; i<features.rows;i++)
    {
        if(labels.at<int>(i,0)==OBJECT)
        {
            objectsCount++;
        }
        else
        {
            if(labels.at<int>(i,0)==NOT_OBJECT)
            {
                negativeCount++;
            }
        }
        if(svm->predict(features.row(i))==labels.at<int>(i,0))
        {
            if(labels.at<int>(i,0)==OBJECT)
            {
                truePositive++;
            }
            else
            {
                if(labels.at<int>(i,0)==NOT_OBJECT)
                {
                    trueNegative++;
                }
            }
        }
    }
    cout<<"Objects classification accuracy:"<<truePositive/objectsCount<<endl;
    cout<<"Negative classification accuracy:"<<trueNegative/negativeCount<<endl;
    cout<<"Common classification accuracy:"<<(truePositive+trueNegative)/(objectsCount+negativeCount);

    cout<<"Count of features:"<<objectsCount+negativeCount<<endl;
}

int main(int argc, char *argv[])
{
    FileStorage config;
    int returnCode=parseCommandLine(argc, argv, config);

    if(returnCode!=ReturnCode::Success)
    {
        return returnCode;
    }

    //load svm size
    Size filterSize;
    config["filter_size"]>>filterSize;

    //load positie samples
    string objects_feature_filename;
    config["objects_feature"]>>objects_feature_filename;
    vector<FeatureMap> objectsFeature;
    vector<int> objectsLabel;
    loadFeature(objects_feature_filename, OBJECT, filterSize, objectsFeature, objectsLabel);

    //load negative samples
    string negative_feature_filename;
    config["negtive_feature"]>>negative_feature_filename;
    vector<FeatureMap> negativeFeature;
    vector<int> negativeLabel;
    loadFeature(negative_feature_filename, NOT_OBJECT, filterSize, negativeFeature, negativeLabel);

    //load all fddb data
    FDDBContainer data;
    string train_data_filename;
    string train_data_folder;
    config["train_data"]>>train_data_filename;
    config["train_data_folder"]>>train_data_folder;
    data.load(train_data_filename, train_data_folder);

    //train
    FeatureMapSVM svm(filterSize);
    svm.train(objectsFeature, negativeFeature);

    DeepPyramid pyramid(config);

    int iterationCount;
    config["max_iter"]>>iterationCount;
    int imageInIteration;
    config["image_in_iter"]>>imageInIteration;
    for(int iter=0;iter<iterationCount;iter++)
    {
        for(int j=0; j<imageInIteration;j++)
        {
            string image_path;
            Mat image;
            vector<Rect> objects;

            data.next(image_path, image, objects);
            vector<FeatureMap> maps;
            pyramid.extractNotObjectsFeatureMap(image, objects, filterSize, maps);
            for(unsigned int m=0;m<maps.size();m++)
            {
                if(svm.predict(maps[m], false)==OBJECT)
                {
                    negativeFeature.push_back(maps[m]);
                }
            }
        }

        //new train
        svm.train(objectsFeature, negativeFeature);

        svm.printAccuracy(objectsFeature, negativeFeature);

        //save
        svm.save("linear_svm"+to_string(iter)+".xml");
    }

    config.release();

    return ReturnCode::Success;
}
