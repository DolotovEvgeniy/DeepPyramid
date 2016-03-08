#include "fddb_container.h"

#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

Rect ellipseToRect(int major_axis_radius,int minor_axis_radius,double angle,int center_x,int center_y)
{
    double alpha,betta;
    alpha=atan(-major_axis_radius*tan(angle)/minor_axis_radius);
    betta=atan(major_axis_radius/(tan(angle)*minor_axis_radius));
    double xMax=center_x+minor_axis_radius*cos(alpha)*cos(angle);
    xMax-=major_axis_radius*sin(alpha)*sin(angle);
    double xMin=center_x-minor_axis_radius*cos(alpha)*cos(angle);
    xMin+=major_axis_radius*sin(alpha)*sin(angle);
    double yMax=center_y+major_axis_radius*sin(betta)*cos(angle);
    yMax+=minor_axis_radius*cos(betta)*sin(angle);
    double yMin=center_y-major_axis_radius*sin(betta)*cos(angle);
    yMin-=minor_axis_radius*cos(betta)*sin(angle);
    int xSide=fabs(xMax-xMin);
    int ySide=fabs(yMin-yMax);

    return Rect(Point(center_x-ySide/2.0,center_y-xSide/2.0),Size(ySide,xSide));
}

Rect readEllipseToRect(istream& file)
{
    double a, b, h, k, phi;
    file>>a;
    file>>b;
    file>>phi;
    file>>h;
    file>>k;
    int type;
    file>>type;

    return ellipseToRect(a,b,phi,h,k);
}

void FDDBContainer::load(string fddb_file, string image_prefix)
{
    ifstream file(fddb_file);

    if(file.is_open()==false)
    {
        std::cerr << "FDDB file '" << fddb_file
                  << "' not found. Exiting" << std::endl;
        return;
    }
    string img_path;
    while(file>>img_path)
    {
        Mat image;
        image=imread(image_prefix+img_path+".jpg");

        if(!image.data)
        {
            std::cerr << "File '" << image_prefix+img_path+".jpg"
                      << "' not found. Exiting." << std::endl;
            return;
        }

        int objectCount;
        file>>objectCount;

        vector<Rect> objects;
        vector<float> confidence;
        for(int i=0;i<objectCount;i++)
        {
            Rect  rect=readEllipseToRect(file);
            objects.push_back(rect);
            confidence.push_back(0);
        }
        imagesPath.push_back(image_prefix+img_path+".jpg");
        objectsList.push_back(objects);
        confidenceList.push_back(confidence);
    }

    resetCounter();
}

void FDDBContainer::increaseCounter()
{
    counter=(counter+1)%imagesPath.size();
}

void FDDBContainer::resetCounter()
{
    counter=0;
}

void FDDBContainer::next(Mat& img, vector<Rect>& objects, vector<float>& confidence)
{
    img=imread(imagesPath[counter]);

    for(unsigned int i=0;i<objects.size();i++)
    {
        objects.push_back(objectsList[counter][i]);
        confidence.push_back(confidenceList[counter][i]);
    }
    increaseCounter();
}

void FDDBContainer::add(const string image_path, const vector<BoundingBox> boxes)
{
    imagesPath.push_back(image_path);
    vector<Rect> objects;
    vector<float> confidence;
    for(int i=0; i<boxes.size();i++)
    {
        objects.push_back(boxes[i].originalImageBox);
        confidence.push_back(boxes[i].confidence);
    }
    objectsList.push_back(objects);
    confidenceList.push_back(confidence);
}

void FDDBContainer::save(string fddb_file)
{
    ofstream file(fddb_file);

    if(file.is_open()==false)
    {
        std::cerr << "Output file '" << fddb_file
                  << "' not created. Exiting" << std::endl;
        return;
    }
    for(unsigned int i=0;i<imagesPath.size();i++)
    {
        file<<imagesPath[i]<<endl;
        file<<objectsList[i].size()<<endl;
        for(unsigned int j=0;j<objectsList[i].size();j++)
        {
            file<<objectsList[i][j].x<<" ";
            file<<objectsList[i][j].y<<" ";
            file<<objectsList[i][j].width<<" ";
            file<<objectsList[i][j].height<<" ";
            file<<confidenceList[i][j]<<endl;
        }
    }
}
