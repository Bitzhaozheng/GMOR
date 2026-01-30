#!/bin/bash
./RegistrationFactory_demo "$@" \
    --alsologtostderr \
    --exp=true \
    --datatype=KITTI \
    --knn=10 \
    --df=0.05 \
    --rot_near_z=true \
    --topk=1 \
    --voxel_size=0.30 \
    --noise_bound=0.60 \
    --radius_normal=0.90 \
    --radius_feature=2.40
