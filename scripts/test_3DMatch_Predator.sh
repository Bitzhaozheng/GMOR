#!/bin/bash
./RegistrationFactory_demo "$@" \
    --alsologtostderr \
    --exp=true \
    --feature=Predator \
    --datatype=3DLoMatch \
    --knn=30 \
    --df=0.05 \
    --topk=12 \
    --noise_bound=0.05
