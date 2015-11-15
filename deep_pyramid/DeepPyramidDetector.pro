#-------------------------------------------------
#
# Project created by QtCreator 2015-11-14T16:30:36
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = DeepPyramidDetector
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

MAIN=train.cpp

SOURCES += \
    nms.cpp \
    deep_pyramid.cpp \
    $$MAIN

LIBS += -L/usr/local/lib \
-lopencv_core \
-lopencv_imgproc \
-lopencv_highgui \
-lopencv_ml \
-lopencv_video \
-lopencv_features2d \
-lopencv_calib3d \
-lopencv_objdetect \
-lopencv_contrib \
-lopencv_legacy \
-lopencv_flann

INCLUDEPATH += /home/evgeniy/caffe/include

LIBS += -L/home/evgeniy/caffe/build/lib \
-lcaffe

INCLUDEPATH += /usr/local/cuda-7.0/include

INCLUDEPATH += /home/evgeniy/caffe/build/include

LIBS += -L/usr/local/lib \
-lglog \

HEADERS += \
    nms.h \
    deep_pyramid.h
