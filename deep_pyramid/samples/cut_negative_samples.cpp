// Copyright 2016 Dolotov Evgeniy

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>

#include <deep_pyramid.h>
#include <fddb_container.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    string pyramid_configuration=argv[1];
    string fddb_file=argv[2];
    string fddb_image_folder=argv[3];
    string feature_prefix=argv[4];
    int negativeCount=atoi(argv[5]);
    int width=atoi(argv[6]);
    int height=atoi(argv[7]);

    FDDBContainer data;
    data.load(fddb_file, fddb_image_folder);

    FileStorage config(pyramid_configuration, FileStorage::READ);
    DeepPyramid pyramid(config);

    int objectCount=0;
    for(int i=0;i<data.size();i++)
    {
        if(objectCount>negativeCount)
        {
            break;
        }
        string image_path;
        Mat image;
        vector<Rect> objects;

        data.next(image_path, image, objects);

        vector<FeatureMap> maps;
        pyramid.extractNotObjectsFeatureMap(image, objects, Size(width, height), maps);
        for(unsigned int j=0;j<maps.size();j++)
        {
            maps[j].save(feature_prefix+std::to_string((long long unsigned int)objectCount)+".xml");
            objectCount++;
        }
    }
    return 0;
}
