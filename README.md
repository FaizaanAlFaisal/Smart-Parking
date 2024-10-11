# Smart-Parking

This project is designed to be an all-encompassing solution for a smart parking. 

The first component is a script that allows easy (manual) labelling of parking spots with a selection-tool.

The second component automatically picks up those labelled parking spots from a stored location, and detects whether a vehicle is parked in those spaces or not.

This allows a simple and interactive way of determining free spots in a parking area.

## How to use:

Start off by running the select-regions.py script, and providing an image input (a screenshot/still frame) from the video feed of a parking lot. Then, follow the instructions and draw in the bounding regions for the parking spots.

Next, run the smart-parking.py script after modifying the parameters per specific needs, and see the program automatically detecting which spaces are free/occupied.