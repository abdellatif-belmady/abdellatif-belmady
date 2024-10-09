---
comments: true
---

# **Change Root File System to SD Card Directly**

In this guide, we will explain how to move your root file system on EMMC flash to SD card storage directly.

## **How to format the SD card as ext4**

First, connect your SD card to SD card slot and connect the basic interfaces (Ethernet, HDMI, keyboard, mouse) then power on.

Open a terminal and type these commands below:

```bash
sudo jetson_clocks
gnome-disks
```

![image](assets/img0.avif)

The first command allows the Jetson module’s whole sources to use. The next command opens GNOME Disks application below.

![image](assets/img1.avif)

Format the whole disk before creating the storage.

![image](assets/img2.avif)

![image](assets/img3.avif)

![image](assets/img4.avif)

Then, create a new partition from SD card storage.

![image](assets/img5.avif)

![image](assets/img6.avif)

Format the disk as ext4 format (partition size is up to you but must be min current file system’s size).

![image](assets/img7.avif)

After creating the partition, check it’s name (/dev/mmcblk1p1).

![image](assets/img8.avif)

## **How to copy the root file system?**

Download the script file from [here](https://github.com/mistelektronik/forecr_blog_files/raw/master/change_rootfs_storage_direct-emmc_to_sdmmc.zip) and extract it. Then, run it with this command below:

```bash
sudo ./change_rootfs_storage_direct-emmc_to_sdmmc.sh {EXTERNAL_STORAGE}
```

In our setup, we typed this command below:

```bash
sudo ./change_rootfs_storage_direct-emmc_to_sdmmc.sh /dev/mmcblk1p1
```

![image](assets/img9.avif)

A few times later, the whole file system copied and the root path changed. 

![image](assets/img10.avif)

It’s time to reboot the Jetson module. Reboot it and check the Root File System copied successfully.

## **How to assign SD card as root file system?**

Open a terminal and type this command to check the root mounted from SD card below:

```bash
df -h
```

![image](assets/img11.avif)

After rebooting you can see that the new storage is assigned as root file system.