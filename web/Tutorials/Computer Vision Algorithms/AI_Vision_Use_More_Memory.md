---
comments: true
---

# **Jetson Nano – Use More Memory!**

## **Introduction**

The NVIDIA Jetson Nano Developer Kit has 4 GB of main memory. This may not enough memory for running many of the large deep learning models, or compiling very large programs. Let’s fix that! We are going to install a swapfile.

## **Installation**

On the [JetsonHacksNano account on Github](https://github.com/JetsonHacksNano), there is a repository named [installSwapFile](https://github.com/JetsonHacksNano/installSwapfile). Clone the repository, and then switch over to the repository directory:

```bash
git clone https://github.com/JetsonHacksNano/installSwapfile
```

```bash
cd installSwapfile
```

Run :

```bash
./installSwapfile.sh
```

and a 6 GB swapfile will be installed at /mnt/swapfile.

!!! warning

	You will need to have enough extra room on your device to install the swapfile.
		
For the 4 GB Jetson Nano, the 6 GB swapfile is what Ubuntu recommends assuming that you are using hibernation. Otherwise 2 GB should do.

Reboot the system so the changes take effect :

```bash
sudo reboot
```