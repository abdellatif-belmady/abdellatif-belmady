---
comments: true
---

# **Jetson-inference Setup Guide**

## **Introduction**

This guide will walk you through the process of setting up and compiling the Jetson Inference project, which is a collection of tools and libraries for real-time video analytics on NVIDIA Jetson platforms.

[https://github.com/dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)

You have two options to setting up the Jetson Inference Project:

- [x] **Option 1:** Run the Docker Container

- [x] **Option 2:** Build the project from source

## **Run the Docker Container**

First, you should clone the files in the project.

```bash
git clone --recursive https://github.com/dusty-nv/jetson-inference
```

By going into jetson-inference directory that created, you must run the container.


```bash
cd jetson-inference 
```

```bash
docker/run.sh 
```

Docker container will automatically run and pull all the files, it will take few minutes depending on the network. This is the first setup and will only be done once. 

Then, you must build the container.

```bash
docker/build.sh 
```

Then you are good.


## **Build the project from source**

### **Before starting**

Before you begin, the following packages have to be installed:

- Git: For cloning the repository.
- CMake: For building the project.
- Python3 and Python3-dev: For building the Python bindings.
- Numpy: A library for the Python programming language, adding support for arrays and matrices.

First, update your package list:

```bash
sudo apt-get update
```

Then, install Git and CMake:

```bash
sudo apt-get install git cmake
```

Install the necessary development packages:

```bash
sudo apt-get install libpython3-dev python3-numpy
```

### **Clone the Repository**

Navigate to your chosen directory and clone the project:

```bash
git clone https://github.com/dusty-nv/jetson-inference
```

By going into jetson-inference directory that created, you must run the container.

```bash
cd jetson-inference
```

```bash
git submodule update –init 
```

Next, to download all the necessary files, and build the project create a folder called ``build`` and run cmake. 

```bash
cd jetson-inference
```

```bash
mkdir build 
```

```bash
cd build 
```

```bash
cmake ../  
```

Then, Model-Downloader tool will run automatically on the screen. This project comes with various pre-trained network models, you can choose which one(s) to download.

![image](assets/img12.avif)

You can also re-run Model-Downloader tool later using the following command.

```bash
cd jetson-inference/tools
```

```bash
./download-models.sh 
```

Then, PyTorch Installer will appear on the screen. PyTorch is used to re-train networks and we will not need it in this project, so you can skip this part.

![image](assets/img13.avif)

To compile the project at the end, run the following commands while in the build directory:

```bash
cd jetson-inference/build
```

```bash
make 
```

```bash
sudo make install 
```

```bash
sudo ldconfig
```

Then we are good.

## **Run the project**

After uccessfully setting up the Jetson Inference project, you can now start using it by executing the command:

```bash
docker/run.sh 
```

To exist, just execute the command:

```bash
exit
```


And that's it! You have successfully set up and compiled the Jetson Inference project. You can now use the provided tools and libraries for real-time video analytics on your NVIDIA Jetson platform.