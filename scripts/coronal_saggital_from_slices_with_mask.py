import argparse
import os
import sys
import pydicom
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
from scipy.ndimage import label
import imageio


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis=1)


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)


def get_3d_bounding_boxes(mask):
    """
    Get 3D bounding boxes for connected components in the mask.
    Each bbox is (zmin, ymin, xmin, zmax, ymax, xmax)
    """
    labeled_mask, num_features = label(mask)
    bboxes = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_mask == i)
        zmin, ymin, xmin = coords.min(axis=0)
        zmin -= 1
        ymin -= 1
        xmin -= 1
        zmax, ymax, xmax = coords.max(axis=0)
        zmax += 1
        ymax += 1
        xmax += 1
        bboxes.append((zmin, ymin, xmin, zmax, ymax, xmax))
    return bboxes


def main():
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument('directory', type=str, help='Path to the slice directory')
    parser.add_argument('segmentation', type=str, help='Path to the segmentation file')
    parser.add_argument('result', type=str, help='Path to the result')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory.")
        sys.exit(1)

    if not os.path.isfile(args.segmentation):
        print(f"Error: '{args.segmentation}' is not a valid file.")
        sys.exit(1)

    slices_directory = args.directory

    os.makedirs(args.result, exist_ok=True)

    # load dicom slice files
    slices = [pydicom.dcmread(os.path.join(slices_directory, f)) 
            for f in os.listdir(slices_directory) if f.endswith('.dcm')]

    # sort slices by SliceLocation (or InstanceNumber)
    slices.sort(key=lambda x: int(x.SliceLocation))

    # stack into 3D volume
    img_dcm = np.stack([s.pixel_array for s in slices], axis=0)

    # load tumor mask
    segm_ds = pydicom.dcmread(args.segmentation)
    tumor = np.zeros_like(img_dcm)

    for i, frame in enumerate(segm_ds.PerFrameFunctionalGroupsSequence):
        # print(int(frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedFrameNumber))
        # copy the tumor slice into the same shaped tumor matrix to rotate
        index = int(frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedFrameNumber)
        # tumor[index - 1, ...] = np.where(segm_ds.pixel_array[i, ...] == 1, img_dcm[index - 1, ...], 0)
        tumor[index - 1, ...] = segm_ds.pixel_array[i, ...]

    # get dcm pixel length in mm to rotate
    pixel_len_mm = [float(slices[0].SliceThickness), float(slices[0].PixelSpacing[0]), slices[0].PixelSpacing[1]]

    img_min = np.amin(img_dcm)
    img_max = np.amax(img_dcm)

    mask_min = np.amin(tumor)
    mask_max = np.amax(tumor)
    cm = matplotlib.colormaps['bone']
    fig, ax = plt.subplots()
    
    # create projection
    n = 24  # TODO: set as argument
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):

        # rotate and show the original image
        rotated_img = rotate_on_axial_plane(img_dcm, alpha)
        projection = MIP_sagittal_plane(rotated_img)
        plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])

        # Rotate and show the tumor mask
        rotated_mask = rotate_on_axial_plane(tumor, alpha)
        projection_mask = MIP_sagittal_plane(rotated_mask)
        plt.imshow(projection_mask, cmap='Reds', alpha=0.5, vmin=mask_min, vmax=mask_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])

        bboxes = get_3d_bounding_boxes(rotated_mask)
        # Clear previous bounding boxes
        for patch in plt.gca().patches:
            patch.remove()

        # Draw bounding boxes on the projection mask
        for zmin, ymin, xmin, zmax, ymax, xmax in bboxes:
            plt.gca().add_patch(plt.Rectangle((ymin, zmin), ymax - ymin, zmax - zmin, edgecolor='blue', facecolor='none', linewidth=1.5))

        projection_file = os.path.join(args.result, f'Projection_{idx}_mask.png')

        plt.savefig(projection_file)
        projections.append(projection)

    

    # Get the list of PNG files in the folder
    png_files = [os.path.join(args.result, f'Projection_{idx}_mask.png') for idx in range(n)]

    # Create a GIF from the PNG files
    gif_path = os.path.join(args.result, 'Projection_with_mask.gif')
    with imageio.get_writer(gif_path, mode='I', duration=0.25, loop=0) as writer:
        for png_file in png_files:
            image = imageio.imread(png_file)
            writer.append_data(image)

    print(f"GIF saved at {gif_path}")

    labeled, num_features = scipy.ndimage.label(tumor)
    centroids = scipy.ndimage.center_of_mass(tumor, labeled, range(1, num_features + 1))
    print("Centroids:")
    for centroid in centroids:
        print(">>", centroid)


if __name__ == '__main__':
    main()
