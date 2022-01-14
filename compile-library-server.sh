
echo CMake
mkdir build
cd build
cmake -DTORCH_PATH=~/anaconda3/envs/py38torch18/lib/python3.8/site-packages/torch -DRENDERER_BUILD_GUI=OFF -DRENDERER_BUILD_TESTS=OFF -DRENDERER_BUILD_CLI=OFF -DRENDERER_BUILD_TESTS=OFF -DRENDERER_BUILD_OPENGL_SUPPORT=OFF ..
make -j8 VERBOSE=true
cd ..

echo Setup-Tools build
python setup.py build
cp build/lib.linux-x86_64-3.8/pyrenderer.cpython-38-x86_64-linux-gnu.so bin/

echo Test
cd bin
python -c "import torch; print(torch.__version__); import pyrenderer; print('pyrenderer imported')"
