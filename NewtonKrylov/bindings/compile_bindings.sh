CPLUS_INCLUDE_PATH=../include/ c++ -O3 -shared -std=c++11 -fPIC    \
-I ../include/pybind11              \
-I ../include/eigen3                \
-I NewtonKrylov ../*.cpp            \
`python3-config --cflags --ldflags` \
bindings.cpp -o NewtonKrylov.so
