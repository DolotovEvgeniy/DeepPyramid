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


class TrainConfiguration
{
public:
    string file_with_train_image;
    string train_image_folder;
    Size filterSize;
    int bootStrapTrainIter;
    int sampleMaxCount;
    string SVMSaveName;

    TrainConfiguration(FileStorage& config)
    {
        config["FileWithTrainImage"]>>file_with_train_image;
        config["TrainImageFolder"]>>train_image_folder;
        config["TrainFilter-size"]>>filterSize;
        config["BootStrapTrainIter"]>>bootStrapTrainIter;
        config["SampleMaxCount"]>>sampleMaxCount;
        config["SVMSaveName"]>>SVMSaveName;
    }

};
#define MARGIN_THRESHOLD 1

Rect originalRect2Norm5(const Rect& originalRect, int level, const Size& imgSize,
                        const Size& featureMapSize, const int& levelCount,
                        const double& levelScale)
{
    double longSide=std::max(imgSize.height, imgSize.width);
    double scale=featureMapSize.width/(pow(levelScale, levelCount-level-1)*longSide);
    return scaleRect(originalRect, scale);
}

int chooseLevel(const Size& filterSize, const Rect& boundBox, const Size& imgSize,
                const Size& featureMapSize, const int& levelCount, const double& levelScale)
{
    vector<double> f;
    for(int i=0;i<levelCount;i++)
    {
        Rect r=originalRect2Norm5(boundBox, i, imgSize, featureMapSize, levelCount, levelScale);

        f.push_back(abs(filterSize.width-r.width)+abs(r.height-filterSize.height));
    }
    int bestLevel=distance(f.begin(), min_element(f.begin(), f.end()));

    return bestLevel;
}

void extractFeatureVectors(const Mat& img, const Size& filterSize, const vector<Rect>& objects,
                           const DeepPyramid& pyramid, Mat& features, Mat& labels)
{
    int stride=1;
    vector<FeatureMap> maps;
    pyramid.constructFeatureMapPyramid(img, maps);
    Size mapSize=maps[0].size();
    for(int level=0; level<pyramid.levelCount; level++)
    {
        vector<Rect> rectAtLevel;
        for(unsigned int rect=0; rect<objects.size();rect++)
        {
            rectAtLevel.push_back(originalRect2Norm5(objects[rect], level,
                                                     Size(img.cols, img.rows),maps[level].size(),
                                                     pyramid.levelCount, pyramid.levelScale));
        }
        for(int width=0; width<mapSize.width-filterSize.width; width+=stride)
            for(int height=0; height<mapSize.height-filterSize.height; height+=stride)
            {
                Rect featureMapRectangle(Point(width, height), filterSize);

                bool extractFeature=true;
                for(unsigned int rect=0;rect<rectAtLevel.size();rect++)
                {
                    if(IOU(featureMapRectangle, rectAtLevel[rect])>0.3)
                    {
                        extractFeature=false;
                        break;
                    }
                }

                if(extractFeature)
                {
                    FeatureMap map;
                    maps[level].extractFeatureMap(featureMapRectangle, map);
                    Mat feature;
                    map.reshapeToVector(feature);
                    features.push_back(feature);
                    labels.push_back(NOT_OBJECT);
                }
            }
    }
    for(unsigned int rect=0;rect<objects.size();rect++)
    {
        int level=chooseLevel(filterSize, objects[rect], Size(img.cols, img.rows),
                              maps[0].size(), pyramid.levelCount, pyramid.levelScale);
        Rect featureMapRectangle=originalRect2Norm5(objects[rect], level,
                                                    Size(img.cols, img.rows),maps[level].size(),
                                                    pyramid.levelCount, pyramid.levelScale);
        FeatureMap map;
        maps[level].extractFeatureMap(featureMapRectangle, map);
        map.resize(filterSize);
        Mat feature;
        map.reshapeToVector(feature);
        features.push_back(feature);
        labels.push_back(OBJECT);
    }
}

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");

    if (configFileName.empty() == true)
    {
        std::cerr << "Configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }

    FileStorage config(configFileName, FileStorage::READ);

    if(config.isOpened()==false)
    {
        std::cerr << "File '" << configFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ConfigFileNotFound;
    }

    DeepPyramid pyramid(config);
    TrainConfiguration trainConfig(config);

    FDDBContainer data;
    data.load(trainConfig.file_with_train_image, trainConfig.train_image_folder);

    Mat features;
    Mat labels;

    for(int i=0;i<30;i++)
    {
        Mat image;
        vector<Rect> objects;
        vector<float> confidence;
        data.next(image, objects, confidence);

        extractFeatureVectors(image, trainConfig.filterSize, objects, pyramid, features, labels);
        cout<<"Count of features:"<<features.rows<<endl;
    }
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);
    CvSVM* svm=new CvSVM();
    svm->train_auto(features,labels, Mat(),Mat(), params);

    for(int i=0;i<trainConfig.bootStrapTrainIter;i++)
    {

        Mat featuresWithOutEasy;
        Mat labelsWithOutEasy;
        for(int i=0;i<features.rows;i++)
        {
            if(labels.at<int>(i,0)==OBJECT || svm->predict(features.row(i))!=labels.at<int>(i,0))
            {
                featuresWithOutEasy.push_back(features.row(i));
                labelsWithOutEasy.push_back(labels.at<int>(i,0));
            }
            else
            {
                if(fabs(svm->predict(features.row(i), true))<MARGIN_THRESHOLD)
                {
                    featuresWithOutEasy.push_back(features.row(i));
                    labelsWithOutEasy.push_back(labels.at<int>(i,0));
                }
            }
        }
        featuresWithOutEasy.copyTo(features);
        labelsWithOutEasy.copyTo(labels);

        featuresWithOutEasy.release();
        labelsWithOutEasy.release();


        Mat newFeatures;
        Mat newLabels;

        Mat image;
        vector<Rect> objects;
vector<float> confidence;
        data.next(image, objects, confidence);

        extractFeatureVectors(image, trainConfig.filterSize, objects, pyramid, newFeatures, newLabels);

        for(int i=0;i<newFeatures.rows;i++)
        {
            if(newLabels.at<int>(i,0)==OBJECT || svm->predict(newFeatures.row(i))!=newLabels.at<int>(i,0))
            {
                features.push_back(newFeatures.row(i));
                labels.push_back(newLabels.at<int>(i,0));
            }
            else
            {
                if(fabs(svm->predict(newFeatures.row(i), true))<MARGIN_THRESHOLD)
                {
                    features.push_back(newFeatures.row(i));
                    labels.push_back(newLabels.at<int>(i,0));
                }
            }
        }

        cout<<"Count of features:"<<features.rows<<endl;

        svm->train_auto(features,labels, Mat(),Mat(), params);
        svm->save(("linear_svm"+std::to_string(static_cast<long long>(i))+".xml").c_str());
    }

    config.release();

    delete svm;
    return ReturnCode::Success;
}
