# Getting Started
You'll need to run this with a Docker capable host computer, installation support is provided 
on Ubuntu.

`install-docker.sh` -- this will get Docker up and running

Now you can get a container ready with:
```
docker build --tag keras-cpu .
docker run -p 8888:8888 --volume $(pwd):/src keras-cpu
```



## GPU Support
`gpu/install-nvidia-docker.sh` -- this requires you have a NVIDIA graphics card as well as the current driver.

If all is well at this point, you will see an inventory of your graphics cards such as:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.77                 Driver Version: 390.77                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN V             Off  | 00000000:03:00.0 Off |                  N/A |
| 37%   53C    P0    38W / 250W |      0MiB / 12066MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  TITAN V             Off  | 00000000:04:00.0 Off |                  N/A |
| 33%   48C    P0    37W / 250W |      0MiB / 12066MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Now you can get a container ready with:
```
docker build --tag keras-gpu ./gpu
nvidia-docker run -p 8888:8888 --volume $(pwd):/src keras-gpu
```