#!/bin/bash
./RegistrationFactory_demo "$@" \
    --alsologtostderr \
    --exp=true \
    --feature=FPFH \
    --datatype=3DMatch \
    --knn=40 \
    --df=0.01 \
    --topk=12 \
    --voxel_size=0.05 \
    --noise_bound=0.10 \
    --radius_normal=0.10 \
    --radius_feature=0.30
