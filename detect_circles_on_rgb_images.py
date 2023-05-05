import numpy as np
import pyopencl as pycl
from pyopencl import array
from PIL import Image, ImageFilter, ImageOps
import argparse


def detect_circles(rgb_image_path: str, num_thetas: int = 120,
                   accumulator_threshold: float = 0.9, pixel_threshold: int = 10) -> None:
    """
    Function to perform Hough circle detection on the input RGB image
    and to print the X center, Y center and diameter of each detected circles.
    :param pixel_threshold: range of pixels, the function removes the circle duplicates
    :param rgb_image_path: file location of the input RGB image
    :param accumulator_threshold: percentage of votes should be given by the
    accumulator to detect a circle with (X,Y) center
    :param num_thetas: number of steps during the rotation
    """

    # Initialize default parameters of minimum and maximum circle radius's
    min_radius = 25
    max_radius = 100

    # Initialize threshold of accumulator votes for circle detection
    accumulator_threshold_votes = int(num_thetas * accumulator_threshold)

    # Initialize step size within the circles and rotation angles as 'thetas'
    delta_theta = int(360 / num_thetas)
    thetas = np.arange(0, 360, step=delta_theta)

    # Definition of sin and cos of rotation angles
    cos_thetas = np.cos(np.deg2rad(thetas))
    cos_thetas = cos_thetas.astype(np.float32)
    sin_thetas = np.sin(np.deg2rad(thetas))
    sin_thetas = sin_thetas.astype(np.float32)

    # Definition of PyOpencl context and command queue
    context = pycl.create_some_context()
    queue = pycl.CommandQueue(context)

    # Preprocessing of input rgb image:
    # 1: Open image and convert ot grayscale
    in_image = Image.open(rgb_image_path)
    image = in_image.convert(mode='L')
    # 2. Perform edge detection and binarize on grayscale image
    image = image.filter(ImageFilter.GaussianBlur)
    image = image.filter(ImageFilter.FIND_EDGES)

    # 3. Convert to np array
    image = np.asarray(image, dtype=np.int32)
    width, height = image.shape

    # Assign arrays to opencl command queue
    image = array.to_device(queue, image)
    cos_thetas = array.to_device(queue, cos_thetas)
    sin_thetas = array.to_device(queue, sin_thetas)

    # Initialize empty arrays for detected circles and accumulator
    circles = array.empty(queue, (width * height, 3), dtype=np.int32)
    accumulator = array.empty(queue, (width, height, (max_radius - min_radius)), dtype=np.int32)

    # Build PyOpenCL program to parallelize circle detection
    program = pycl.Program(context, """
    __kernel void detect_circles(__global const int *image, __global int *circles, __global int *accumulator, __global const float *sin_thetas, 
    __global const float *cos_thetas, int min_radius, int max_radius, int num_thetas, int accumulator_threshold)
    {
      const int x = get_global_id(1);
      const int y = get_global_id(0);
      
      const int width = get_global_size(1);
      const int height = get_global_size(0);
      for(int r = min_radius; r < max_radius; r++)
      {
            for(int t = 0; t < num_thetas; t++)
            {
                if(image[x + y * width] != 0)
                {
                    const int x_center = x + r * cos_thetas[t];
                    const int y_center = y + r * sin_thetas[t];
                    if (x_center >= 0 && x_center < width &&  y_center >= 0 && y_center < height)
                    {
                        const int acc_idx = x_center + y_center * width + (r - min_radius) * width * height;
                        accumulator[acc_idx] += 1;
                        if(accumulator[acc_idx] > accumulator_threshold)
                        {
                            circles[x_center * y_center * 3] = x_center;
                            circles[x_center * y_center * 3 + 1] = y_center;
                            circles[x_center * y_center * 3 + 2] = r * 2;
                        }
                    }
                }
            }
      }
    }""").build()

    # Run PyOpenCL program
    program.detect_circles(queue, image.shape, None, image.data, circles.data, accumulator.data, sin_thetas.data,
                           cos_thetas.data, np.int32(min_radius),
                           np.int32(max_radius), np.int32(num_thetas), np.int32(accumulator_threshold_votes))

    # Get and store detected circles
    detected_circles = np.array(circles.get())
    detected_circles = detected_circles[detected_circles[:, 0] != 0]

    # Remove circle duplicates within the given pixel threshold
    final_circles = []
    for x, y, r in detected_circles:
        if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold
               or abs(r - rc) > pixel_threshold * 2 for xc, yc, rc in final_circles):
            final_circles.append((x, y, r))

    # Print detected circles
    for i in range(len(final_circles)):
        print(f'X center: {final_circles[i][0]}, Y center: {final_circles[i][1]}, Diameter: {final_circles[i][2]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--Path', help="path for the input RGB Image", default='examples/coins.jpg')
    parser.add_argument('-n', '--NumThetas', help="step number of circle", default=120)
    parser.add_argument('-t', '--Threshold', help="ratio of votes to detect circles", default=0.9)
    parser.add_argument('-pt', '--PixelThreshold', help="pixel threshold to remove circle duplicates", default=10)
    args = parser.parse_args()
    detect_circles(args.Path, int(args.NumThetas), float(args.Threshold), int(args.PixelThreshold))

