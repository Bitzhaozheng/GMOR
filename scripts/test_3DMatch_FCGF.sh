#!/bin/bash
./RegistrationFactory_demo "$@" \
    --alsologtostderr \
    --exp=true \
    --feature=FCGF \
    --knn=60 \
    --df=0.07 \
    --topk=12 \
    --noise_bound=0.10
