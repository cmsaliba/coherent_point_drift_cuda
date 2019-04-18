# coherent_point_drift_cuda
[![DOI](https://zenodo.org/badge/175826952.svg)](https://zenodo.org/badge/latestdoi/175826952)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Introduction
This is a CUDA implementation of the Coherent Point Drift (CPD) algorithm for non-rigid point set registration. Given two point sets **X** and **Y** the algorithm finds the transform that aligns **Y** to **X**. The aligned point set is **T = Y + GW**.

**CUDA Implementation (latest release)**: Chris Saliba. (2019, April 18). Coherent Point Drift CUDA (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.2646522

**Coherent Point Drift Algorithm**: Myronenko, A., & Song, X. (2010). Point set registration: Coherent point drift. IEEE transactions on pattern analysis and machine intelligence, 32(12), 2262-2275.

## Requirements
* CUDA capable GPU
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone) Toolkit and Driver
* [CLI11](https://github.com/CLIUtils/CLI11) Command line interpreter - `CLI11.hpp` is included in the repository and is governed by its associated license
* C/C++ compiler, version must be compatible with CUDA
    * Windows - [MSVC](https://www.visualstudio.com/)
    * Linux - [GNU compiler](https://gcc.gnu.org/)
    * refer to [CUDA documentation](http://docs.nvidia.com/cuda/) for compatible compiler versions for your platform
* [CMake](https://cmake.org/) cross-platform build system

## CMake Targets
* **cpd_cuda**: Static C++ library containing the CUDA implementation of the CPD algorithm.
* **cpd_cmd**: Command line program. Links against cpd_cuda.
* **cpd_mex**: MEX wrapper around the cpd_cuda library for use with MATLAB.

## Windows Builds
Tested using CUDA 9.0 and CUDA 10.1 using Visual Studio 2015. For the command-line build you can avoid a full installation of the Visual Studio IDE and only install the Visual Studio Build Tools.

### Command Line
1. Open a command prompt window and set up the Visual Studio development tools. The path to bat file will depend on your version of Visual Studio. The `amd64` argument specifies a 64bit target.
```
C:\>"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
```
2. Navigate to the root *coherent_point_drift_cuda* directory.
```
C:\>cd path_to_cpd_cuda
```
3. Perform an out of source build. 
    * (Optional) Set the `CMAKE_BUILD_TYPE` (default is release).
    * Set the `CPD_CUDA_GENCODE` option to match the compute capability of your NVIDIA GPU (default is `-gencode arch=compute_50,code=sm_50` for compute capability 5.0). 
    * (Optional) Set the `CPD_CUDA_BUILD_CMD` and/or `CPD_CUDA_BUILD_MEX` options to TRUE (default FALSE) to build the command-line program and/or MEX wrapper. 
```
path_to_cpd_cuda>mkdir build
path_to_cpd_cuda>cd build
path_to_cpd_cuda/build> cmake -G "NMake Makefiles" -DCPD_CUDA_GENCODE:STRING="-gencode arch=compute_61,code=sm_61" -DCPD_CUDA_BUILD_CMD:BOOL=TRUE -DCPD_CUDA_BUILD_MEX:BOOL=TRUE ..
cpd_cuda/build> nmake
``` 
4. The *cpd_cuda.lib* static library is output to *cpd_cuda/lib* and the corresponding header *cpd_cuda.h* is in *cpd_cuda/src*. The command-line executable and the MEX function are output to *cpd_cuda/bin*.

### GUI & Visual Studio
1. Open the CMake GUI and set the source code location to *path_to_cpd_cuda* and the build location to *path_to_cuda/build*.
2. Press Configure and select your version of Visual Studio (for the example this was Visual Studio 14 2015 win64) and Use Default Native Compilers. Press Finish.
3. Change configuration options:
    * CPD_CUDA_GENCODE: change to match the compute capability of your NVIDIA GPU (default is -gencode arch=compute_50,code=sm_50 for compute capability 5.0).
    * (Optional) CPD_CUDA_BUILD_CMD and CPD_CUDA_BUILD_MEX: select to buld the example command line program and/or MEX wrapper.
4. Press Configure.
5. Press Generate.
6. Open the generated Visual Studio solution, select the desired configuration, and build the desired targets.
7. The *cpd_cuda.lib* static library is output to *cpd_cuda/lib* and the corresponding header *cpd_cuda.h* is in *cpd_cuda/src*. The command-line executable and the MEX function are output to *cpd_cuda/bin*.

## Linux Build (Command Line)
Tested on 64-bit Ubuntu 16.04 with version 5.4 of the GNU compilers.

1. Open a terminal and navigate to the root *coherent_point_drift_cuda* directory.
```
C:\>cd path_to_cpd_cuda
```
3. Perform an out of source build. 
    * (Optional) Set the `CMAKE_BUILD_TYPE` (default is release).
    * Set the `CPD_CUDA_GENCODE` option to match the compute capability of your hardware (default is `-gencode arch=compute_50,code=sm_50` for compute capability 5.0). 
    * (Optional) Set the `CPD_CUDA_BUILD_CMD` and/or `CPD_CUDA_BUILD_MEX` options to TRUE (default FALSE) to build the command-line program and/or MEX wrapper. 
```
path_to_cpd_cuda>mkdir build
path_to_cpd_cuda>cd build
path_to_cpd_cuda/build> cmake -G "Unix Makefiles" -DCPD_CUDA_GENCODE:STRING="-gencode arch=compute_61,code=sm_61" -DCPD_CUDA_BUILD_CMD:BOOL=TRUE ..
cpd_cuda/build> make
``` 
4. The *cpd_cuda.a* static library is output to *cpd_cuda/lib* and the corresponding header *cpd_cuda.h* is in *cpd_cuda/src*. The command-line executable is output to *cpd_cuda/bin*.

## Usage

### C++ Static Library
Include the *cpd_cuda.h* header file and link against the *cpd_cuda.a* or *cpd_cuda.lib* library. The CoherentPointDrift function is overloaded to use single or double precision. Note that double precision performance is significantly slower on most GPUs.

```
void CoherentPointDrift(float *Xraw, float *Yraw, int M, int N, int D, float w, float beta, float lambda, float tol, int max_iter, float *Traw, int *C)

void CoherentPointDrift(double *Xraw, double *Yraw, int M, int N, int D,double w, double beta, double lambda, double tol, int max_iter, double *Traw, int *C)
```

**Input**

* `Xraw`: reference point set, stored column-major [N x D]
* `Yraw`: point set to transform, stored column-major [M x D]
* `M`, `N`: number of points in Yraw and Xraw respectively
* `D`: dimensionality of the point sets
* `w`, beta, lambda: optimization parameters
* `tol`: tolerance stop criteria
* `max_iter`: maximum number of iterations to run

**Output**

* `Traw`: transformed point set, stored column-major [M x D]
* `C`: correspondence vector [M x 1], only calculated if `C != NULL`

**Command Line**   
With the *cpd_cmd* executable in yourcurrent directory.
```
Usage: ./cpd_cmd [OPTIONS] xpts_file ypts_file tpts_file [correspondence_file]

Positionals:
  xpts_file TEXT REQUIRED     File containing the target point set.
  ypts_file TEXT REQUIRED     File containing the point set to transform.
  tpts_file TEXT REQUIRED     File to write the transformed point set to.
  correspondence_file TEXT    File to write the correspondence vector to.

Options:
  -h,--help                   Print this help message and exit
  --use_double_precision      Use double precision math.
  --w FLOAT=0.1               Point set noise parameter (0 <= w <= 1).
  --beta FLOAT=2              Smoothness and regularization parameter (beta > 0).
  --lambda FLOAT=3            Smoothness and regularization parameter (lambda > 0).
  --tol FLOAT=1e-05           Convergence tolerance (tol >= 0).
  --max_iter INT=500          Maximum iterations to run.
```

* The only required inputs are `xpts`, `ypts`, and `tpts`. 
* `xpts` and `ypts` are the input files containing the X and y point sets. 
* `tpts` is the file that will be written out containing the transformed points.
* If `correspondence_file` is provided, a file will be written out with the correspondence vector between the point sets.
* All other algorithm parameters will fall back to the default values if not provided.
* `use_double_precision` should only be set to `true` if you are using a GPU with good double precision performance (i.e.: a Tesla card).

### Matlab/MEX 
The function *cpd_cuda.m* wraps the mex function in an easy-to-use interface. Example scripts and files are also provided in the *mat* subdirectory. You will need to add the mex function to your path or move it to your working directory after it has been built.

## Notes
* The `CoherentPointDrift` functions can use single precision or double precision for calculations. On most GPUs double precision performance is significantly slower than single precision and single precision is recommended.
* When the NVIDIA GPU running the CUDA code is also driving a display, the display driver can sometimes timeout causing the CUDA kernels to crash, or the display to freeze. The reason for this is the length of time required to solve the system of equations on the GPU. The kernel is still running, but the driver watchdog timer flags it as unresponsive. The workaround is to plug the display into the integrated graphics output or another graphics card. Alternatively, you can can boot directly into a terminal session, and run the command line program from there. 
