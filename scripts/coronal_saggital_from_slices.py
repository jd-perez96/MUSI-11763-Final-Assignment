import argparse
import os
import sys
import pydicom
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis=1)


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)


def main():
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument('directory', type=str, help='Path to the slice directory')
    parser.add_argument('result', type=str, help='Path to the result')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory.")
        sys.exit(1)

    print(f"Directory received: {args.directory}")
    slices_directory = args.directory

    os.makedirs(args.result, exist_ok=True)

    # load dicom slice files
    slices = [pydicom.dcmread(os.path.join(slices_directory, f)) 
            for f in os.listdir(slices_directory) if f.endswith('.dcm')]

    # sort slices by SliceLocation (or InstanceNumber)
    slices.sort(key=lambda x: int(x.SliceLocation))

    # stack into 3D volume
    img_dcm = np.stack([s.pixel_array for s in slices], axis=0)

    # get dcm pixel length in mm to rotate
    pixel_len_mm = [float(slices[0].SliceThickness), float(slices[0].PixelSpacing[0]), slices[0].PixelSpacing[1]]

    img_min = np.amin(img_dcm)
    img_max = np.amax(img_dcm)
    cm = matplotlib.colormaps['bone']
    fig, ax = plt.subplots()
    
    # create projection
    n = 24  # TODO: set as argument
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        rotated_img = rotate_on_axial_plane(img_dcm, alpha)
        projection = MIP_sagittal_plane(rotated_img)
        plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        projection_file = os.path.join(args.result, f'Projection_{idx}.png')
        plt.savefig(projection_file)
        projections.append(projection)

    # save animation
    animation_data = [
        [plt.imshow(img, animated=True, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                interval=250, blit=True)
    anim.save(os.path.join(args.result, 'animation.gif'))  # Save animation


if __name__ == '__main__':
    main()
