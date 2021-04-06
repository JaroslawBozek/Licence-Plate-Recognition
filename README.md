# Licence-Plate-Recognition
![](img/plate.jpg)
## Goal
The goal of the project was to create a script that is able to read Polish licence plates using OpenCV library
## Assumptions
* Horizontal and vertical angle of the presented licence plate may vary by +- 45 degrees
* Longer side of licence plate is at least 1/3 size of the image
* All licence plates have 7 characters
* The resolution of images may vary

## Requirements
* Python 3.7
* Processing time of each image may not extend 2*images seconds (Specifications of the testing PC were unknown but it was pretty modern for the time of making this project)
* All libraries were available.
* In case of using machine learning, the model must be trained using the available calculation time. (The project was tested on around 50 images)

## Results
Private Set (49 images):
* Detected characters 99.41% (341/343)
* Properly recognised characters 96.79% (332/343)
* Overall score 93.46% (458/490)

Unknown Set (~50 images):
* Overall score 78.91%

Score system:
* +1 point per recognised character
* +3 points if all 7 characters were recognised
* The character is only counted if it's placed in a right place in the output string

## How does it work?

![](img/54m7dp.gif)
