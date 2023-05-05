# HoughCircleDetection
Command line application to perform Hough circle detection on RGB images

Description of command line parameters:

-p: Path to the input RGB image. e.g. -p 'examples/coins.jpg'

-n: Number of theta rotation angles during circle detection. e.g. -n 120

-t: Accumulator threshold - Ratio of votes that should be given by the accumulator to detect the circle

-pt: Pixel threshold - Within this range, the function keeps only one circle in order to remove circle duplicates
