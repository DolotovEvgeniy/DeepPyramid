#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/tools/caffe train --solver=examples/ddfd_alexnet/solver.prototxt --weights=models/bvlc_alexnet/bvlc_alexnet.caffemodel --gpu 0 2>&1 | tee examples/ddfd_alexnet/log
