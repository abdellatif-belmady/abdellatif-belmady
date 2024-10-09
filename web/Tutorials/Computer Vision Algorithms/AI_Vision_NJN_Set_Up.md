---
comments: true
---

# **Jetson Nano Preparation Guide**

This guide will walk you through the process of initializing and preparing your Jetson Nano for usage.

## **Prerequisites**
- Jetson Nano
- Mouse, Keyboard, Screen, and a 5V power adapter (included)
- 32GB micro SD card
- Balena Etcher: [balenaEtcher](https://etcher.balena.io/)
- SD Formatter: [SD Formatter](https://www.sdcard.org/downloads/formatter/sd-memory-card-formatter-for-windows-download/)
- Jetson Nano SDK: [Jetson Nano SDK img](https://developer.nvidia.com/jetson-nano-sd-card-image)

## **Steps to Prepare Jetson Nano**
1. **Format the SD Card**: Insert the SD card into your computer and format it using the SD Formatter tool.

2. **Flash the Jetson SDK**: Use the Balena Etcher tool to flash the Jetson Nano SDK into the SD card.

3. **Insert the SD Card into Jetson Nano**: Safely eject the SD card from your computer and insert it into the Jetson Nano.

4. **Connect Peripherals**: Connect the mouse, keyboard, and screen to the Jetson Nano.

5. **Power On**: Plug the power adapter into the Jetson Nano. It should power on automatically.

6. **Accept EULA**: Upon booting, accept the NVIDIA Jetson software End User License Agreement (EULA).

7. **Configure System**: Select your system language, keyboard layout, and time zone.

8. **Create User Account**: Create a username, password, and computer name for your Jetson Nano.

9. **Allocate APP Partition Size**: Select the maximum recommended size for the APP partition.

10. **First Commands**: 
    For best practices, begin by executing these commands:

    ```bash
    sudo apt update
    ```

    ```bash
    sudo apt install python3-pip
    ```

    ```bash
    sudo pip3 install -U jetson-stats
    ```

    To ensure the changes take effect, you need to reboot the system:

    ```bash
    sudo reboot
    ```

    ```bash
    jtop
    ```

By following these steps, you will have successfully initialized and prepared your Jetson Nano for usage.