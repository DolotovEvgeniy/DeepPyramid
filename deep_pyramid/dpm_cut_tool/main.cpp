#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>
#include <deep_pyramid.h>

using namespace cv;
using namespace std;
using namespace caffe;
Point point1, point2; /* vertical points of the bounding box */
int drag = 0;
Rect rect; /* bounding box */
Mat img, roiImg; /* roiImg - the part of the image in the bounding box */
int select_flag = 0;

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
    }

    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        /* mouse dragged. ROI being selected */
        Mat img1 = img.clone();
        point2 = Point(x, y);
        rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
        imshow("image", img1);
    }

    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point(x, y);
        rect = Rect(point1.x,point1.y,x-point1.x,y-point1.y);
        drag = 0;
        roiImg = img(rect);
        select_flag = 1;
    }

    if (event == CV_EVENT_LBUTTONUP)
    {
       /* ROI selected */
        select_flag = 1;
        drag = 0;
    }
}

int chooseLevel(Size filterSize, Rect boundBox, Mat& img)
{
    double f[7];
    for(int i=0;i<7;i++)
    {

        Rect r=getNorm5RectByOriginal(boundingBoxAtLevel(i,boundBox,Size(img.cols,img.rows)));

        f[i]=abs(filterSize.width-r.width)+abs(r.height-filterSize.height);
    }
    double minVal=f[0];
    int min=0;
    for(int i=0;i<7;i++)
    {
        if(minVal>f[i])
        {
            minVal=f[i];
            min=i;
        }
    }
    return min;
}

int main(int argc, char *argv[])
{   
    Caffe::set_mode(Caffe::CPU);

    string alexnet_model_file=argv[1];
    string alexnet_trained_file=argv[2];
    DeepPyramid pyramid(7,alexnet_model_file, alexnet_trained_file);
    string image_file;
    Mat features;
    Mat label;
    int pos=1,neg=1;
    for(int pp=0;pp<1;pp++)
    {
        cout<<"Image path:";
        cin>>image_file;
        cout<<endl;
        img=imread(image_file, CV_LOAD_IMAGE_COLOR);

        pyramid.setImg(img);
        pyramid.createImagePyramid();
        pyramid.createMax5Pyramid();
        pyramid.createNorm5Pyramid();
        for(int kk=0;kk<20;kk++)
        {
            cout<<"Phase: Positive"<<endl;
            cout<<"Choose face and press SPACE"<<endl;
            imshow("image", img);
            cvSetMouseCallback("image", mouseHandler, NULL);
            waitKey(0);
            imshow("Positive", roiImg); /* show the image bounded by the box */
            cout<<rect.size()<<endl;

            Rect boundBox=rect;
            int level=chooseLevel(Size(7,11),boundBox,img);
            cout<<"Level: "<<level<<endl;
            Mat feature(1,11*7*256,CV_32FC1);
            for(int i=0;i<256;i++)
            {
                Mat resized;
                resize(pyramid.norm5[level][i](getNorm5RectByOriginal(boundingBoxAtLevel(level,boundBox,Size(img.cols,img.rows)))),resized,Size(7,11));
                for(int w=0;w<7;w++)
                    for(int h=0;h<11;h++)
                    {
                        feature.at<float>(0,w+h*7+i*11*7)=resized.at<float>(h,w);
                    }
            }
            features.push_back(feature);
            cout<<"Face?(Y-1,N-0)"<<endl;
            int key;
            cin>>key;
            if(key==1)
            {
                label.push_back(1);
                cout<<"face"<<endl;
                pos++;
            }
            if(key==0)
            {
                label.push_back(-1);
                cout<<"not face"<<endl;
                neg++;
            }
        }

        destroyWindow("Positive");
        destroyWindow("image");
        waitKey(0);
    }
    //cout<<features.cols<<endl<<label.cols<<endl;
    CvSVM svm;
    svm.train(features,label);
    svm.save("svm.xml");
  //  pyramid.createImagePyramid();
   // pyramid.createMax5Pyramid();
   // pyramid.createNorm5Pyramid();



    return 0;
}
