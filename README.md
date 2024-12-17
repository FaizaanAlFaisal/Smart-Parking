# Smart-Parking

This project is designed to be a simple solution for a smart parking system. 

The first component is a script that allows easy (but manual) labelling of parking spots with a selection-tool.

The second component automatically picks up those labelled parking spots from a stored location, and detects whether a vehicle is parked in those spaces or not.

This allows a simple and interactive way of determining free spots in a parking area.

## Demo Video

![Demo video](assets/ParkingDemo.gif)

## How to use:

First setup the environment, and install requirements. Ensure PyTorch is installed with CUDA/GPU enabled for faster processing.

Start off by running the `select-regions.py` script, and providing an image input (a screenshot/still frame) from the video feed of a parking lot. Then, follow the instructions and draw in the bounding regions for the parking spots.

Next, run the `smart-parking.py` script after modifying the parameters as per specific needs, and see the program automatically detecting which spaces are free/occupied. Works with live feed or a video file input.

## Requirements

This project was developed with the following setup:

- Python 3.11+
- PyTorch 2.4.1 with CUDA 12.4