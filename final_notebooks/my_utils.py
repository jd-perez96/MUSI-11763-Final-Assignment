import os
import imageio
import pydicom
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import label


def load_image(directory):
    """Load DICOM image from a directory and sort its slices by `SliceLocation`"""
    slices = [
        pydicom.dcmread(os.path.join(directory, file))
        for file in os.listdir(directory)
        if file.endswith('.dcm')
    ]
    slices.sort(key=lambda x: int(x.SliceLocation))
    return slices


def assert_acquisition_number(img):
    """Assert if the image have onle one acquisition number by `AcquisitionNumber`"""
    if len(set([item.AcquisitionNumber for item in img])) == 1:
        print("Same acquisition number")
    else:
        print("Different acquisition number")


def assert_geometric_properties(img):
    if len(set([x.SliceThickness for x in img])) == 1:
        print("Same slice thickness")
    else:
        print("Different slice thickness")
    
    if len(set([x.PixelSpacing[0] for x in img])) == 1:
        print("Same pixel spacing at x-axis")
    else:
        print("Different pixel spacing at x-axis")

    if len(set([x.PixelSpacing[1] for x in img])) == 1:
        print("Same pixel spacing at y-axis")
    else:
        print("Different pixel spacing at y-axis")


def print_geometric_properties(img):
    """Print geometric properties from DICOM image list"""
    print("Pixel Spacing, x-axis:  ", img[0].PixelSpacing[0])
    print("Pixel Spacing, y-axis:  ", img[0].PixelSpacing[1])
    print("Slice Thickness, z-axis:", img[0].SliceThickness)

    increment = set([abs(A.ImagePositionPatient[2] - B.ImagePositionPatient[2]) for A, B in zip(img[:-1], img[1:])])
    print("Slice Increment:", increment)

    height, width = img[0].pixel_array.shape
    print("3D volume (x, y, z) in mm:", img[0].PixelSpacing[0] * width, img[0].PixelSpacing[0] * height, img[0].SliceThickness * len(img))


def make_axial_plane_gif(img, title, path="../results/"):
    # get min-max values
    all_pixels = np.concatenate([x.pixel_array.flatten() for x in img])
    min_val = all_pixels.min()
    max_val = all_pixels.max()

    # plot middle frame
    index = len(img) // 2
    plt.title(f"Axial plane frame {index}")
    plt.axis('off')
    plt.imshow(img[index].pixel_array, cmap='bone', vmin=min_val, vmax=max_val)
    plt.show()

    # make animation
    fig_anim, ax_anim = plt.subplots()
    ax_anim.axis('off')

    def update(frame):
        # min_val = img[frame].pixel_array.min()
        # max_val = img[frame].pixel_array.max()
        ax_anim.clear()
        ax_anim.axis('off')
        ax_anim.set_title(f"Axial plane (frame {frame})")
        return ax_anim.imshow(img[frame].pixel_array, cmap='bone', vmin=min_val, vmax=max_val),

    ani = animation.FuncAnimation(
        fig_anim, update, frames=len(img), blit=True
    )

    img_path = os.path.join(path, title)
    os.makedirs(path, exist_ok=True)
    ani.save(img_path, writer='pillow', fps=10)
    plt.close(fig_anim)
    print("Animation saved on", img_path)


def make_coronal_plane_gif(img, title, path="../results/"):
    pixel_len_mm = [img[0].SliceThickness, img[0].PixelSpacing[0], img[0].PixelSpacing[1]]

    # Stack slices into a 3D numpy array
    volume = np.stack([s.pixel_array for s in img], axis=0)

    # Transpose to (y, z, x) so coronal is along axis 1
    coronal_volume = np.transpose(volume, (1, 0, 2))
    min_val = coronal_volume.min()
    max_val = coronal_volume.max()

    # plot middle slice
    index = coronal_volume.shape[0] // 2
    plt.title(f"Coronal plane frame {index}")
    plt.axis('off')
    plt.imshow(coronal_volume[index], cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.show()

    # make animation
    fig_anim, ax_anim = plt.subplots()
    ax_anim.axis('off')

    def update(frame):
        # min_val = coronal_volume[frame].min()
        # max_val = coronal_volume[frame].max()
        ax_anim.clear()
        ax_anim.axis('off')
        ax_anim.set_title(f"Coronal plane (frame {frame})")
        return ax_anim.imshow(coronal_volume[frame], cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1]),

    ani = animation.FuncAnimation(
        fig_anim, update, frames=len(coronal_volume), blit=True
    )

    img_path = os.path.join(path, title)
    os.makedirs(path, exist_ok=True)
    ani.save(img_path, writer='pillow', fps=10)
    plt.close(fig_anim)
    print("Animation saved on", img_path)


def make_saggital_plane_gif(img, title, path="../results/"):
    # get min-max values
    all_pixels = np.concatenate([x.pixel_array.flatten() for x in img])
    min_val = all_pixels.min()
    max_val = all_pixels.max()

    pixel_len_mm = [img[0].SliceThickness, img[0].PixelSpacing[0], img[0].PixelSpacing[1]]

    # Stack slices into a 3D numpy array
    volume = np.stack([s.pixel_array for s in img], axis=0)

    # Transpose to (y, z, x) so coronal is along axis 1
    sagittal_volume = np.transpose(volume, (2, 0, 1))

    # plot middle slice
    index = sagittal_volume.shape[0] // 2
    plt.title(f"Sagittal plane frame {index}")
    plt.axis('off')
    plt.imshow(sagittal_volume[index], cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.show()

    # make animation
    fig_anim, ax_anim = plt.subplots()
    ax_anim.axis('off')

    def update(frame):
        min_val = sagittal_volume[frame].min()
        max_val = sagittal_volume[frame].max()
        ax_anim.clear()
        ax_anim.axis('off')
        ax_anim.set_title(f"Saggital plane (frame {frame})")
        return ax_anim.imshow(sagittal_volume[frame], cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1]),

    ani = animation.FuncAnimation(
        fig_anim, update, frames=len(sagittal_volume), blit=True
    )

    img_path = os.path.join(path, title)
    os.makedirs(path, exist_ok=True)
    ani.save(img_path, writer='pillow', fps=10)
    plt.close(fig_anim)
    print("Animation saved on", img_path)


def median_sagittal_plane(img_dcm):
    """ Compute the median sagittal plane of the CT image provided. """
    return img_dcm[:, :, img_dcm.shape[1]//2]


def median_coronal_plane(img_dcm):
    """ Compute the median sagittal plane of the CT image provided. """
    return img_dcm[:, img_dcm.shape[2]//2, :]


def median_axial_plane(img_dcm):
    return img_dcm[img_dcm.shape[0]//2, :, :]


def MIP_sagittal_plane(img_dcm):
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm):
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis=1)


def MIP_axial_plane(img_dcm):
    return np.max(img_dcm, axis=0)


def AIP_sagittal_plane(img_dcm):
    """ Compute the average intensity projection on the sagittal orientation. """
    return np.mean(img_dcm, axis=2)


def AIP_coronal_plane(img_dcm):
    """ Compute the average intensity projection on the coronal orientation. """
    return np.mean(img_dcm, axis=1)


def AIP_axial_plane(img_dcm):
    return np.mean(img_dcm, axis=0)


def plot_planes(img, func_axial, func_coronal, func_sagittal, title):
    pixel_len_mm = [
        img[0].SliceThickness,
        img[0].PixelSpacing[0],
        img[0].PixelSpacing[1]
    ]

    all_pixels = np.concatenate([x.pixel_array.flatten() for x in img])
    min_val = all_pixels.min()
    max_val = all_pixels.max()

    volume = np.stack([s.pixel_array for s in img])

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)

    median_data = func_axial(volume)
    min_val = median_data.min()
    max_val = median_data.max()
    axs[0, 0].imshow(median_data, cmap='bone', vmin=min_val, vmax=max_val)
    axs[0, 0].set_title('Axial plane')
    axs[0, 0].axis('off')

    median_data = func_sagittal(volume)
    min_val = median_data.min()
    max_val = median_data.max()
    axs[0, 1].imshow(median_data, cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    axs[0, 1].set_title('Saggital plane')
    axs[0, 1].axis('off')

    median_data = func_coronal(volume)
    min_val = median_data.min()
    max_val = median_data.max()
    axs[1, 0].imshow(median_data, cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    axs[1, 0].set_title('Coronal plane')
    axs[1, 0].axis('off')

    axs[1, 1].axis('off')  # Empty subplot

    plt.tight_layout()
    plt.show()


def rotate_on_axial_plane(img_dcm, angle_in_degrees, order, cval):
    return scipy.ndimage.rotate(
        img_dcm, angle_in_degrees, axes=(1, 2), reshape=False, mode='constant',
        order=order, cval=cval)


def coronal_sagittal_rotation(img, title, n=24, path='../results/MIP/'):
    # Create projections varying the angle of rotation
    #   Configure visualization colormap

    pixel_len_mm = [
        img[0].SliceThickness,
        img[0].PixelSpacing[0],
        img[0].PixelSpacing[1]
    ]

    img_dcm = np.stack([s.pixel_array for s in img], axis=0)
    fig, ax = plt.subplots()
    #   Configure directory to save results
    os.makedirs(path, exist_ok=True)

    #   Create projections
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        rotated_img = rotate_on_axial_plane(img_dcm, alpha, order=3, cval=img_dcm.min())
        projection = MIP_coronal_plane(rotated_img)

        img_min = np.amin(projection)
        img_max = np.amax(projection)

        plt.title("Coronal-Sagittal rotation")
        plt.axis('off')
        plt.imshow(projection, cmap='bone', vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.savefig(os.path.join(path, f"Projection_{idx}.png"))
        projections.append(projection)

    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, cmap='bone', vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
        for img in projections
    ]

    anim = animation.ArtistAnimation(fig, animation_data, interval=250, blit=True)
    img_path = os.path.join(path, title)
    anim.save(img_path, writer='pillow')
    plt.show()

    print("Animation saved on", img_path)


def make_axial_plane_with_seg_gif(img, seg, liver, title, path="../results/"):
    # get min-max values
    all_pixels = np.concatenate([x.pixel_array.flatten() for x in img])
    min_val = all_pixels.min()
    max_val = all_pixels.max()

    # plot middle frame
    index = 13
    plt.title(f"Axial plane frame {index}")
    plt.axis('off')
    plt.imshow(img[index].pixel_array, cmap='bone', vmin=min_val, vmax=max_val)
    plt.imshow(np.ma.masked_where(liver[index] == 0, liver[index]), cmap='Greens', vmin=0, vmax=1, alpha=0.4)
    plt.imshow(np.ma.masked_where(seg[index] == 0, seg[index]), cmap='Reds', vmin=0, vmax=1, alpha=0.4)
    plt.show()

    # make animation
    fig_anim, ax_anim = plt.subplots()
    ax_anim.axis('off')

    def update(frame):
        # min_val = img[frame].pixel_array.min()
        # max_val = img[frame].pixel_array.max()
        ax_anim.clear()
        ax_anim.axis('off')
        ax_anim.set_title(f"Axial plane (frame {frame})")
        ax_anim.imshow(img[frame].pixel_array, cmap='bone', vmin=min_val, vmax=max_val)
        ax_anim.imshow(np.ma.masked_where(liver[frame] == 0, liver[frame]), cmap='Greens', vmin=0, vmax=1, alpha=0.4)
        return ax_anim.imshow(np.ma.masked_where(seg[frame] == 0, seg[frame]), cmap='Reds', vmin=0, vmax=1, alpha=0.4),

    ani = animation.FuncAnimation(
        fig_anim, update, frames=len(img), blit=True
    )

    img_path = os.path.join(path, title)
    os.makedirs(path, exist_ok=True)
    ani.save(img_path, writer='pillow', fps=10)
    plt.close(fig_anim)
    print("Animation saved on", img_path)


def make_coronal_plane_with_seg_gif(img, seg, liver, title, path="../results/"):
    pixel_len_mm = [img[0].SliceThickness, img[0].PixelSpacing[0], img[0].PixelSpacing[1]]

    # Stack slices into a 3D numpy array
    volume = np.stack([s.pixel_array for s in img], axis=0)

    # Transpose to (y, z, x) so coronal is along axis 1
    coronal_volume = np.transpose(volume, (1, 0, 2))
    seg = np.transpose(seg, (1, 0, 2))
    liver = np.transpose(liver, (1, 0, 2))

    min_val = coronal_volume.min()
    max_val = coronal_volume.max()

    # plot middle frame
    index = coronal_volume.shape[0] // 2
    plt.title(f"Coronal plane frame {index}")
    plt.axis('off')
    plt.imshow(coronal_volume[index], cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.imshow(np.ma.masked_where(liver[index] == 0, liver[index]), cmap='Greens', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.imshow(np.ma.masked_where(seg[index] == 0, seg[index]), cmap='Reds', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.show()

    # make animation
    fig_anim, ax_anim = plt.subplots()
    ax_anim.axis('off')

    def update(frame):
        # min_val = coronal_volume[frame].min()
        # max_val = coronal_volume[frame].max()
        ax_anim.clear()
        ax_anim.axis('off')
        ax_anim.set_title(f"Coronal plane (frame {frame})")
        ax_anim.imshow(coronal_volume[frame], cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        ax_anim.imshow(np.ma.masked_where(liver[frame] == 0, liver[frame]), cmap='Greens', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        return ax_anim.imshow(np.ma.masked_where(seg[frame] == 0, seg[frame]), cmap='Reds', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1]),

    ani = animation.FuncAnimation(
        fig_anim, update, frames=len(coronal_volume), blit=True
    )

    img_path = os.path.join(path, title)
    os.makedirs(path, exist_ok=True)
    ani.save(img_path, writer='pillow', fps=10)
    plt.close(fig_anim)
    print("Animation saved on", img_path)


def make_sagittal_plane_with_seg_gif(img, seg, liver, title, path="../results/"):
    pixel_len_mm = [img[0].SliceThickness, img[0].PixelSpacing[0], img[0].PixelSpacing[1]]

    # Stack slices into a 3D numpy array
    volume = np.stack([s.pixel_array for s in img], axis=0)

    # Transpose to (y, z, x) so coronal is along axis 1
    sagittal_volume = np.transpose(volume, (2, 0, 1))
    seg = np.transpose(seg, (2, 0, 1))
    liver = np.transpose(liver, (2, 0, 1))

    min_val = sagittal_volume.min()
    max_val = sagittal_volume.max()

    # plot middle frame
    index = sagittal_volume.shape[0] // 2
    plt.title(f"Sagittal plane frame {index}")
    plt.axis('off')
    plt.imshow(sagittal_volume[index], cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.imshow(np.ma.masked_where(liver[index] == 0, liver[index]), cmap='Greens', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.imshow(np.ma.masked_where(seg[index] == 0, seg[index]), cmap='Reds', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.show()

    # make animation
    fig_anim, ax_anim = plt.subplots()
    ax_anim.axis('off')

    def update(frame):
        # min_val = coronal_volume[frame].min()
        # max_val = coronal_volume[frame].max()
        ax_anim.clear()
        ax_anim.axis('off')
        ax_anim.set_title(f"Sagittal plane (frame {frame})")
        ax_anim.imshow(sagittal_volume[frame], cmap='bone', vmin=min_val, vmax=max_val, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        ax_anim.imshow(np.ma.masked_where(liver[frame] == 0, liver[frame]), cmap='Greens', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        return ax_anim.imshow(np.ma.masked_where(seg[frame] == 0, seg[frame]), cmap='Reds', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1]),

    ani = animation.FuncAnimation(
        fig_anim, update, frames=len(sagittal_volume), blit=True
    )

    img_path = os.path.join(path, title)
    os.makedirs(path, exist_ok=True)
    ani.save(img_path, writer='pillow', fps=10)
    plt.close(fig_anim)
    print("Animation saved on", img_path)


def get_2d_bounding_boxes(mask):
    labeled_mask, num_features = label(mask)
    bboxes = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_mask == i)
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        bboxes.append((ymin, xmin, ymax, xmax))
    return bboxes


def coronal_sagittal_seg_rotation(img, seg, liver, title, n=24, path='../results/MIP_seg/'):
    pixel_len_mm = [
        img[0].SliceThickness,
        img[0].PixelSpacing[0],
        img[0].PixelSpacing[1]
    ]

    img_dcm = np.stack([s.pixel_array for s in img], axis=0)
    os.makedirs(path, exist_ok=True)

    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        rotated_img = rotate_on_axial_plane(img_dcm, alpha, order=3, cval=img_dcm.min())
        projection = MIP_coronal_plane(rotated_img)

        rotated_mask = rotate_on_axial_plane(seg, alpha, order=0, cval=0)
        #bboxes = get_3d_bounding_boxes(rotated_mask)
        rotated_mask = MIP_coronal_plane(rotated_mask)
        bboxes = get_2d_bounding_boxes(rotated_mask)

        rotated_liver = rotate_on_axial_plane(liver, alpha, order=0, cval=0)
        rotated_liver = MIP_coronal_plane(rotated_liver)

        img_min = projection.min()
        img_max = projection.max()

        plt.axis('off')
        plt.imshow(projection, cmap='bone', vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.imshow(np.ma.masked_where(rotated_liver == 0, rotated_liver), cmap='Greens', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        plt.imshow(np.ma.masked_where(rotated_mask == 0, rotated_mask), cmap='Reds', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])

        for patch in plt.gca().patches:
            patch.remove()

        for ymin, xmin, ymax, xmax in bboxes:
            margin = 1  # expand a bit the bounding box
            x0 = max(xmin - margin, 0)
            y0 = max(ymin - margin, 0)
            x1 = xmax + margin
            y1 = ymax + margin
            plt.gca().add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='yellow',
                              facecolor='none', linewidth=1.5))

        plt.savefig(os.path.join(path, f"Projection_{idx}_seg.png"))

        # ax = plt.gca()
        # ax.axis('off')
        # im1 = ax.imshow(projection, cmap='bone', vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        # im2 = ax.imshow(np.ma.masked_where(rotated_liver == 0, rotated_liver), cmap='Greens', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        # im3 = ax.imshow(np.ma.masked_where(rotated_mask == 0, rotated_mask), cmap='Reds', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        # # Draw bounding box and centroid on top of the segmentation
        # rect = plt.Rectangle(
        #     (bbox[0][2], bbox[0][1]),  # (xmin, ymin)
        #     bbox[1][2] - bbox[0][2],   # width
        #     bbox[1][1] - bbox[0][1],   # height
        #     linewidth=4, edgecolor='cyan', facecolor='none', linestyle='--'
        # )
        # ax.add_patch(rect)
        # # Draw centroid
        # ax.plot(centroid[2], centroid[1], 'yo', markersize=8, markeredgecolor='black')
        # fig = plt.gcf()
        # fig.savefig(os.path.join(path, f"Projection_{idx}_seg.png"))
        # ax.clear()

    # Get the list of PNG files in the folder
    png_files = [os.path.join(path, f'Projection_{idx}_seg.png') for idx in range(n)]

    # Create a GIF from the PNG files
    gif_path = os.path.join(path, "reference_coronal-sagittal_rotation_seg.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.25, loop=0) as writer:
        for png_file in png_files:
            image = imageio.imread(png_file)
            writer.append_data(image)

    print(f"GIF saved at {gif_path}")


def coronal_sagittal_seg_rotation2(img, tumor, liver, segm, title, n=24, path='../results/segmentation/'):
    pixel_len_mm = [
        img[0].SliceThickness,
        img[0].PixelSpacing[0],
        img[0].PixelSpacing[1]
    ]

    img_dcm = np.stack([s.pixel_array for s in img], axis=0)
    os.makedirs(path, exist_ok=True)

    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        rotated_img = rotate_on_axial_plane(img_dcm, alpha, order=3, cval=img_dcm.min())
        projection = MIP_coronal_plane(rotated_img)

        rotated_mask = rotate_on_axial_plane(tumor, alpha, order=0, cval=0)
        rotated_mask = MIP_coronal_plane(rotated_mask)
        # bboxes = get_2d_bounding_boxes(rotated_mask)

        rotated_liver = rotate_on_axial_plane(liver, alpha, order=0, cval=0)
        rotated_liver = MIP_coronal_plane(rotated_liver)

        rotated_segm = rotate_on_axial_plane(segm, alpha, order=0, cval=0)
        rotated_segm = MIP_coronal_plane(rotated_segm)
        bboxes = get_2d_bounding_boxes(rotated_segm)

        img_min = projection.min()
        img_max = projection.max()

        plt.axis('off')
        plt.imshow(projection, cmap='bone', vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.imshow(np.ma.masked_where(rotated_liver == 0, rotated_liver), cmap='Greens', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        plt.imshow(np.ma.masked_where(rotated_mask == 0, rotated_mask), cmap='Reds', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        plt.imshow(np.ma.masked_where(rotated_segm == 0, rotated_segm), cmap='Blues', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])

        for patch in plt.gca().patches:
            patch.remove()

        for ymin, xmin, ymax, xmax in bboxes:
            margin = 1  # expand a bit the bounding box
            x0 = max(xmin - margin, 0)
            y0 = max(ymin - margin, 0)
            x1 = xmax + margin
            y1 = ymax + margin
            plt.gca().add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='yellow',
                              facecolor='none', linewidth=1.5))

        plt.savefig(os.path.join(path, f"Projection_{idx}_seg.png"))

        # ax = plt.gca()
        # ax.axis('off')
        # im1 = ax.imshow(projection, cmap='bone', vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        # im2 = ax.imshow(np.ma.masked_where(rotated_liver == 0, rotated_liver), cmap='Greens', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        # im3 = ax.imshow(np.ma.masked_where(rotated_mask == 0, rotated_mask), cmap='Reds', vmin=0, vmax=1, alpha=0.4, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        # # Draw bounding box and centroid on top of the segmentation
        # rect = plt.Rectangle(
        #     (bbox[0][2], bbox[0][1]),  # (xmin, ymin)
        #     bbox[1][2] - bbox[0][2],   # width
        #     bbox[1][1] - bbox[0][1],   # height
        #     linewidth=4, edgecolor='cyan', facecolor='none', linestyle='--'
        # )
        # ax.add_patch(rect)
        # # Draw centroid
        # ax.plot(centroid[2], centroid[1], 'yo', markersize=8, markeredgecolor='black')
        # fig = plt.gcf()
        # fig.savefig(os.path.join(path, f"Projection_{idx}_seg.png"))
        # ax.clear()

    # Get the list of PNG files in the folder
    png_files = [os.path.join(path, f'Projection_{idx}_seg.png') for idx in range(n)]

    # Create a GIF from the PNG files
    gif_path = os.path.join(path, "reference_coronal-sagittal_rotation_seg.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.25, loop=0) as writer:
        for png_file in png_files:
            image = imageio.imread(png_file)
            writer.append_data(image)

    print(f"GIF saved at {gif_path}")
