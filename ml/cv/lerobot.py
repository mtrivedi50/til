"""
Notes on LeRobot setup on Mac OSX:

1. Install specific version of libtorchcodec, make sure it matches PyTorch version. I
   chose PyTorch 2.8 and torchcodec 0.7.
2. According to Claude, torchcodec only supports up to ffmpeg7
3. Set up DYLD_LIBRARY_PATH in ~/.bashrc. The libavutil.59.dylib torchcodec needs is now
   at that path.
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset