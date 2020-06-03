## Introduction
This project shows backend optimization in SLAM.
## Requirements

### Eigen Package (Version >= 3.0.0)
#### Source
http://eigen.tuxfamily.org/index.php?title=Main_Page

#### Compile and Install
```
cd [path-to-Eigen]
mkdir build
cd build
cmake ..
make 
sudo make install 
```

#### Search Installing Location
```
sudo updatedb
locate eigen3
```

default location "/usr/include/eigen3"

### Sophus Package
#### Download
https://github.com/HugoNip/Sophus

#### Compile and Install
```
cd [path-to-pangolin]
mkdir build
cd build
cmake ..
make 
sudo make install 
```

### BAL dataset
https://grail.cs.washington.edu/projects/bal/


### MeshLab install
```
sudo apt-get update
sudo apt-get install meshlab
```


## Compile this Project
```
mkdir build
cd build
cmake ..
make 
```

## Run
```
./bundle_adjustment_g2o
```

## Result
![Screenshot%20from%202020-06-03%2010-06-00.png](https://github.com/HugoNip/SLAMBackEndOptimization/blob/master/results/Screenshot%20from%202020-06-03%2010-06-00.png)



## Reference
[Source](https://github.com/HugoNip/slambook2/tree/master/ch9)
