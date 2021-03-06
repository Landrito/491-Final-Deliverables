#!/usr/bin/env bash

######################################################################
# Torch install
######################################################################


if [[ `uname` != 'Linux' ]]; then
  echo 'Platform unsupported, only available for Linux'
  exit
fi
if [[ `which apt-get` == '' ]]; then
    echo 'apt-get not found, platform not supported'
    exit
fi

# Install dependencies for Torch:
sudo apt-get update
sudo apt-get install -qqy build-essential
sudo apt-get install -qqy gcc g++
sudo apt-get install -qqy cmake
sudo apt-get install -qqy curl
sudo apt-get install -qqy libreadline-dev
sudo apt-get install -qqy git-core
sudo apt-get install -qqy libjpeg-dev
sudo apt-get install -qqy libpng-dev
sudo apt-get install -qqy ncurses-dev
sudo apt-get install -qqy imagemagick
sudo apt-get install -qqy unzip
sudo apt-get update


echo "==> Torch7's dependencies have been installed"





# Build and install Torch7
cd /tmp
rm -rf luajit-rocks
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir -p build
cd build
git checkout master; git pull
rm -f CMakeCache.txt
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make install
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi


path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then
    cutorch=ok
    cunn=ok
fi

# Install base packages:
luarocks install cwrap
luarocks install paths
luarocks install torch
luarocks install nn

[ -n "$cutorch" ] && \
(luarocks install cutorch)
[ -n "$cunn" ] && \
(luarocks install cunn)

luarocks install luafilesystem
luarocks install penlight
luarocks install sys
luarocks install xlua
luarocks install image
luarocks install env
luarocks install qtlua
luarocks install qttorch

echo ""
echo "=> Torch7 has been installed successfully"
echo ""


echo "Installing nngraph ... "
luarocks install nngraph
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "nngraph installation completed"

echo "Installing Xitari ... "
cd /tmp
rm -rf xitari
git clone https://github.com/deepmind/xitari.git
cd xitari
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Xitari installation completed"

echo "Installing Alewrap ... "
cd /tmp
rm -rf alewrap
git clone https://github.com/deepmind/alewrap.git
cd alewrap
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Alewrap installation completed"

echo
echo "You can run experiments by executing: "
echo
echo "   ./run_cpu game_name"
echo
echo "            or   "
echo
echo "   ./run_gpu game_name"
echo
echo "For this you need to provide the rom files of the respective games (game_name.bin) in the roms/ directory"
echo

