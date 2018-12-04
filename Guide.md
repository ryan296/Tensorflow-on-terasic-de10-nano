# Building TensorFlow for Cyclone V SoC FPGA: a Step-By-Step Guide

_[Back to readme](README.md)_

## What You Need

* Terasic DE10-Nano or similar development kit with Cyclone V SoC FPGA
* An SD card running Linux LXDE Desktop (kernel 4.5) with several GB of free space (A 16 GB SD card minimum is recommended)
* Internet connection to the DE10-Nano
* A USB memory drive that can be installed as swap memory (if it is a flash drive, make sure you don't care about the drive). Anything over 1 GB should be fine
* A fair amount of time

### Contents

1. [Install basic dependencies](#1-install-basic-dependencies)
2. [Install USB Memory as Swap](#2-install-a-memory-drive-as-swap-for-compiling)
3. [Build Bazel](#3-build-bazel)
4. [Compiling TensorFlow](#4-compiling-tensorflow)
5. [Cleaning Up](#5-cleaning-up)
6. [References](#references)

## The Build

### 1. Install basic dependencies

First, update apt-get to make sure it knows where to download everything.

```shell
sudo apt-get update
```

Next, install some base dependencies and tools we'll need later.

For Bazel:

```shell
sudo apt-get install build-essential openjdk-8-jdk pkg-config zip g++ zlib1g-dev unzip
```

For TensorFlow:

```
# For Python 2.7
sudo apt-get install python-pip python-numpy swig python-dev
sudo pip install wheel numpy mock
sudo pip install keras_applications==1.0.5 --no-deps
sudo pip install keras_preprocessing==1.0.3 --no-deps
sudo pip install absl-py astor gast six protobuf tensorboard tensorflow_estimator termcolor

# For Python 3.3+
sudo apt-get install python3-pip python3-numpy swig python3-dev
sudo pip3 install wheel numpy mock
sudo pip3 install keras_applications==1.0.5 --no-deps
sudo pip3 install keras_preprocessing==1.0.3 --no-deps
sudo pip3 install absl-py astor gast six protobuf tensorboard tensorflow_estimator termcolor
```

To be able to take advantage of certain optimization flags:

```
sudo apt-get install gcc-4.8 g++-4.8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100
```

Finally, for cleanliness, make a directory that will hold the Protobuf, Bazel, and TensorFlow repositories.

```shell
mkdir tf
cd tf
```

### 2. Install a Memory Drive as Swap for Compiling

In order to succesfully build TensorFlow, your DE-10 Nano needs a little bit more memory to fall back on. Fortunately, this process is pretty straightforward. Grab a USB storage drive that has at least 1GB of memory. I used a flash drive I could live without that carried no important data. That said, we're only going to be using the drive as swap while we compile, so this process shouldn't do too much damage to a relatively new USB drive.

First, put insert your USB drive, and find the `/dev/XXX` path for the device.

```shell
sudo blkid
```

As an example, my drive's path was `/dev/sda1`

Once you've found your device, unmount it by using the `umount` command.

```shell
sudo umount /dev/XXX
```

Then format your device to be swap:

```shell
sudo mkswap /dev/XXX
```

If the previous command outputted an alphanumeric UUID, copy that now. Otherwise, find the UUID by running `blkid` again. Copy the UUID associated with `/dev/XXX`

```shell
sudo blkid
```

Now edit your `/etc/fstab` file to register your swap file

```shell
sudo nano /etc/fstab
```

On a separate line, enter the following information. Replace the X's with the UUID (without quotes)

```bash
UUID=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX none swap sw,pri=5 0 0
```

Save `/etc/fstab`, exit your text editor, and run the following command:

```shell
sudo swapon -a
```

If you get an error claiming it can't find your UUID, go back and edit `/etc/fstab`. Replace the `UUID=XXX..` bit with the original `/dev/XXX` information.

```shell
sudo nano /etc/fstab
```

```bash
# Replace the UUID with /dev/XXX
/dev/XXX none swap sw,pri=5 0 0
```

Alright! You've got swap! Don't throw out the `/dev/XXX` information yet- you'll need it to remove the device safely later on.

### 3. Build Bazel

To build [Bazel](https://github.com/bazelbuild/bazel), we're going to need to download a zip file containing a distribution archive. Let's do that now and extract it into a new directory called `bazel`:

```shell
wget https://github.com/bazelbuild/bazel/releases/download/0.11.0/bazel-0.11.0-dist.zip
unzip -d bazel bazel-0.11.0-dist.zip
```

Once it's done downloading and extracting, we can move into the directory to make a few changes:

```shell
cd bazel
```

Before building Bazel, we need to set the `javac` maximum heap size for this job, or else we'll get an OutOfMemoryError. To do this, we need to make a small addition to `bazel/scripts/bootstrap/compile.sh`. (Shout-out to @SangManLINUX for [pointing this out.](https://github.com/samjabrahams/tensorflow-on-raspberry-pi/issues/5#issuecomment-210965695).

```shell
nano scripts/bootstrap/compile.sh
```

Move down to line 117, where you'll see the following block of code:

```shell
run "${JAVAC}" -classpath "${classpath}" -sourcepath "${sourcepath}" \
      -d "${output}/classes" -source "$JAVA_VERSION" -target "$JAVA_VERSION" \
      -encoding UTF-8 "@${paramfile}"
```

At the end of this block, add in the `-J-Xmx500M` flag, which sets the maximum size of the Java heap to 500 MB:

```
run "${JAVAC}" -classpath "${classpath}" -sourcepath "${sourcepath}" \
      -d "${output}/classes" -source "$JAVA_VERSION" -target "$JAVA_VERSION" \
      -encoding UTF-8 "@${paramfile}" -J-Xmx500M
```

Finally, we have to add one thing to `tools/cpp/lib_cc_configure.bzl` - open it up for editing:

```shell
nano tools/cpp/lib_cc_configure.bzl
```

Place the line `return "arm"` around line 133 (at the beginning of the `_get_cpu_value` function):

```shell
...
"""Compute the cpu_value based on the OS name."""
return "arm"
...
```

Now we can build Bazel! _Note: this also takes some time._

```shell
sudo ./compile.sh
```

When the build finishes, you end up with a new binary, `output/bazel`. Copy that to your `/usr/local/bin` directory.

```shell
sudo cp output/bazel /usr/local/bin/bazel
```

To make sure it's working properly, run `bazel` on the command line and verify it prints help text. Note: this may take 15-30 seconds to run, so be patient!

```shell
bazel

Usage: bazel <command> <options> ...

Available commands:
  analyze-profile     Analyzes build profile data.
  build               Builds the specified targets.
  canonicalize-flags  Canonicalizes a list of bazel options.
  clean               Removes output files and optionally stops the server.
  dump                Dumps the internal state of the bazel server process.
  fetch               Fetches external repositories that are prerequisites to the targets.
  help                Prints help for commands, or the index.
  info                Displays runtime info about the bazel server.
  mobile-install      Installs targets to mobile devices.
  query               Executes a dependency graph query.
  run                 Runs the specified target.
  shutdown            Stops the bazel server.
  test                Builds and runs the specified test targets.
  version             Prints version information for bazel.

Getting more help:
  bazel help <command>
                   Prints help and options for <command>.
  bazel help startup_options
                   Options for the JVM hosting bazel.
  bazel help target-syntax
                   Explains the syntax for specifying targets.
  bazel help info-keys
                   Displays a list of keys used by the info command.
```

Move out of the `bazel` directory, and we'll move onto the next step.

```shell
cd ..
```

### 4. Compiling TensorFlow

First things first, clone the TensorFlow repository and move into the newly created directory.

```shell
git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

If you're looking to build to a specific version or commit of TensorFlow (as opposed to the HEAD at master), you should `git checkout` it now:

```shell
git checkout v1.9.0
```

Once in the directory, we have to write a nifty one-liner that is incredibly important. The next line goes through all files and changes references of 64-bit program implementations (which we don't have access to) to 32-bit implementations. Neat!

```shell
grep -Rl 'lib64' | xargs sed -i 's/lib64/lib/g'
```

Next, we need to delete a particular line in `tensorflow/core/platform/platform.h`. Open up the file in your favorite text editor:

```shell
sudo nano tensorflow/core/platform/platform.h
```

Now, scroll down toward the bottom and delete the following line containing `#define IS_MOBILE_PLATFORM` (around line 48):

```cpp
#elif defined(__arm__)
#define PLATFORM_POSIX
...
#define IS_MOBILE_PLATFORM   <----- DELETE THIS LINE
```
Now let's configure the build, enter options as entered below :

```
./configure
You have bazel 0.11.0- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3

Found possible Python library paths:
  /usr/local/lib/python3.5/dist-packages
  /usr/lib/python3/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python3.5/dist-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: y
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [y/N]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
        --config=mkl            # Build with MKL support.
        --config=monolithic     # Config for mostly static monolithic build.
        --config=tensorrt       # Build with TensorRT support.
Configuration finished
```
Before building tensor one important step needs to be taken otherwise the resulted python wheel of TensorFlow will give you error when you try to import it inside python and that means you have wasted all of your precious time ! What we are going to do is commenting out some lines according to this [issue](https://github.com/tensorflow/tensorflow/issues/17986) :

```shell
sed -i -e 's/#include "tensorflow\/core\/kernels\/concat_lib.h"/\/\/#include "tensorflow\/core\/kernels\/concat_lib.h"/g' core/kernels/list_kernels.h
sed -i -e 's/#include "tensorflow\/core\/kernels\/concat_lib.h"/\/\/#include "tensorflow\/core\/kernels\/concat_lib.h"/g' core/kernels/list_kernels.cc
sed -i -e 's/ConcatCPU<T>(c->device(), inputs_flat, &output_flat);/\/\/ConcatCPU<T>(c->device(), inputs_flat, &output_flat);/g' core/kernels/list_kernels.h
```
Now we can use ir to build TensorFlow! _Note: This takes almost one day (24 hours) to compile. So, be patient._

```shell
bazel build -c opt --copt="-mfpu=neon" --copt="-funsafe-math-optimizations" --copt="-ftree-vectorize" --copt="-fomit-frame-pointer" --local_resources 1024,1.0,1.0 --verbose_failures tensorflow/tools/pip_package:build_pip_package
```

Now you can finally make a python wheel, issue the following command :

```shell
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```
This one will not take too much time and you can install your python package form `/tmp/tensorflow_pkg` directory :
```shell
pip3 install --user /tmp/tensorflow_pkg/ensorflow-1.6.0-cp35-cp35m-linux_armv7l.whl
```
And your TensorFlow is ready to use.

### 5. Cleaning Up

There's one last bit of house-cleaning we need to do before we're done: remove the USB drive that we've been using as swap.

First, turn off your drive as swap:

```shell
sudo swapoff /dev/XXX
```

Finally, remove the line you wrote in `/etc/fstab` referencing the device

```
sudo nano /etc/fstab
```

Then reboot your DE-10 Nano.

**And you're done!** You deserve a break.

## References

1. [Building TensorFlow for Raspberry Pi: a Step-By-Step Guide](https://github.com/samjabrahams/tensorflow-on-raspberry-pi/blob/master/GUIDE.md)
2. [Building TensorFlow for Orange Pi Zero from source](https://github.com/rezaxdi/tensorflow-on-orangepi-zero/blob/master/GUIDE.md)
3. [Turning a USB Drive into swap](http://askubuntu.com/questions/173676/how-to-make-a-usb-stick-swap-disk)
4. [Safely removing USB swap space](http://askubuntu.com/questions/616437/is-it-safe-to-delete-a-swap-partition-on-a-usb-install)

---

_[Back to top](#building-tensorflow-for-raspberry-pi-a-step-by-step-guide)_
