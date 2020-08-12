#!/bin/bash
set -e
set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"

rm -rf ../buildTEMP
mkdir ../buildTEMP
cd ../buildTEMP
cmake ../BitonicSort -DCMAKE_BUILD_TYPE=Release $@
make