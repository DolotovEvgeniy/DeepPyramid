// Copyright 2016 Dolotov Evgeniy

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "fddb_container.h"

using namespace std;
using namespace cv;

Rect ellipseToRect(int major_axis_radius, int minor_axis_radius,
                   double angle, int center_x, int center_y) {
    double alpha, betta;
    alpha = atan(-major_axis_radius*tan(angle)/minor_axis_radius);
    betta = atan(major_axis_radius/(tan(angle)*minor_axis_radius));
    double xMax = center_x+minor_axis_radius*cos(alpha)*cos(angle);
    xMax-=major_axis_radius*sin(alpha)*sin(angle);
    double xMin = center_x-minor_axis_radius*cos(alpha)*cos(angle);
    xMin+=major_axis_radius*sin(alpha)*sin(angle);
    double yMax = center_y+major_axis_radius*sin(betta)*cos(angle);
    yMax+=minor_axis_radius*cos(betta)*sin(angle);
    double yMin = center_y-major_axis_radius*sin(betta)*cos(angle);
    yMin-=minor_axis_radius*cos(betta)*sin(angle);
    int xSide = fabs(xMax-xMin);
    int ySide = fabs(yMin-yMax);

    return Rect(Point(center_x-ySide/2.0, center_y-xSide/2.0),
                Size(ySide, xSide));
}

Rect readEllipseToRect(ifstream& file) {
    double a, b, h, k, phi;
    file >> a >> b >> phi >> h >> k;
    int type;
    file >> type;

    return ellipseToRect(a, b, phi, h, k);
}

void FDDBContainer::load(string fddb_file, string image_prefix) {
    ifstream file(fddb_file);

    if (file.is_open() == false) {
        std::cerr << "FDDB file '" << fddb_file
                  << "' not found. Exiting" << std::endl;
        return;
    }
    prefix = image_prefix;
    string img_path;
    while (file >> img_path) {
        Mat image;
        image = imread(prefix+img_path+".jpg");

        if (!image.data) {
            std::cerr << "File '" << prefix+img_path+".jpg"
                      << "' not found. Exiting." << std::endl;
            return;
        }

        int objectCount;
        file >> objectCount;

        vector<Rect> objects;
        for (int i = 0; i < objectCount; i++) {
            Rect  rect = readEllipseToRect(file);
            objects.push_back(rect);
        }
        imagesPath.push_back(img_path);
        objectsList.push_back(objects);
    }

    resetCounter();
}

void FDDBContainer::increaseCounter() {
    counter = (counter+1)%imagesPath.size();
}

void FDDBContainer::resetCounter() {
    counter = 0;
}

void FDDBContainer::reset() {
    resetCounter();
}

void FDDBContainer::next(string& image_path, Mat& img, vector<Rect>& objects) {
    image_path = imagesPath[counter];
    img = imread(prefix+imagesPath[counter]+".jpg");

    for (size_t i = 0; i < objectsList[counter].size(); i++) {
        objects.push_back(objectsList[counter][i]);
    }

    increaseCounter();
}

int FDDBContainer::size() {
    return imagesPath.size();
}
