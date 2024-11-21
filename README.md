# sofe4590u-assignment2

## Description

On a Linux VM, I decided to use QEMU to emulate x86_64 and aarch64 CPU architectures for the execution of main.py.

This assignment involves running QEMU to do object detection & classification on a video using a pre-trained ML model on two different CPU architectures.

## Setting up QEMU

First, we made x86_64-rootfs and aarch64-rootfs directories

```bash
mkdir x86_64-rootfs
mkdir aarch64-rootfs
```

Then we set up the root filesystems using debootstrap

```bash
sudo debootstrap --arch=amd64 focal ~/x86_64-rootfs http://archive.ubuntu.com/ubuntu/
sudo debootstrap --arch=arm64 focal ~/aarch64-rootfs http://ports.ubuntu.com/
```

Then it was as simple as using `chroot` to enter the root filesystems and set everything up.

To enter the x86_64-rootfs:

```bash
sudo chroot ~/x86_64-rootfs
```

To enter the aarch64-rootfs:

```bash
sudo chroot ~/aarch64-rootfs
```

For each system, I had to edit the `/etc/apt/sources.list` file to use the correct architecture.

```bash
deb http://ports.ubuntu.com/ubuntu-ports focal main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports focal-updates main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports focal-security main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports focal-backports main restricted universe multiverse
```

For each system, I installed python3 and python3-pip and the necessary pypi packages.

```bash
apt update && apt upgrade
apt install python3 python3-pip
pip3 install opencv-python-headless ultralytics torch torchvision
```

## Running the script

```bash
sudo apt install qemu qemu-user qemu-user-static libgl1
```

I copied the script and the yolov5nu.pt file to the x86_64-rootfs and aarch64-rootfs directories.

```bash
sudo cp main.py x86_64-rootfs/
sudo cp car.mp4 x86_64-rootfs/
sudo cp main.py aarch64-rootfs/
sudo cp car.mp4 aarch64-rootfs/
```

Then I ran the script using QEMU.

```bash
qemu-x86_64 -L x86_64-rootfs/ x86_64-rootfs/usr/bin/python3 x86_64-rootfs/main.py
```

```bash
qemu-aarch64 -L aarch64-rootfs/ aarch64-rootfs/usr/bin/python3 aarch64-rootfs/main.py
```
