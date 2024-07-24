import pandas as pd
import shutil
import os
import glob


def returnPt(pt_time, ground_truth):
    cul = 0
    # Read motion data
    motion_data = pd.read_csv(ground_truth)
    motion_timestamps = motion_data['timestamp'].tolist()  # Convert timestamps to a list
    #  Read image timestamps
    with open(pt_time, 'r') as f:
        image_timestamps = [line.strip() for line in f]
    selected_indices = []  #Store the corresponding ground truth

    # Iterate over each pair of consecutive motion data timestamps
    for i in range(len(motion_timestamps) - 1):
        start = motion_timestamps[i]
        end = motion_timestamps[i + 1]

        # Find image timestamps between the two motion timestamps
        for index, timestamp in enumerate(image_timestamps):
            if start <= int(timestamp) < end:
                selected_indices.append(index)
                break  # Stop after finding the first one. If you need to find all images between the two timestamps, remove this break

    for i in range(len(selected_indices)):
        cul += 1
    # Output the results
    print(cul)
    return selected_indices


def copy_selected_images(numbers_list, source_dir, target_dir):
    """
     Copy the images corresponding to the specified numbers from the source folder to the target folder

    :param numbers_list: A list containing the image numbers
    :param source_dir: The directory containing the source images
    :param target_dir: The target directory where the selected images will be copied to
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Iterate over the list of numbers
    for number in numbers_list:
        # Construct the paths for the source file and the target file
        source_file = os.path.join(source_dir, f"{number}.png")
        target_file = os.path.join(target_dir, f"{number}.png")

        # Check if the source file exists
        if os.path.exists(source_file):
            # copy files
            shutil.copy(source_file, target_file)
            print(f"Copied {source_file} to {target_file}")
        else:
            print(f"File {source_file} not found.")


def clear_images_in_folder(folder_path):
    # Construct the image search pattern, assuming the image extension is .png
    pattern = os.path.join(folder_path, '*.png')

    # Use glob to find all matching image files
    image_files = glob.glob(pattern)

    # Check if there are any image files
    if image_files:
        print(f"find {len(image_files)} images，cleaning the folder...")
        for image_file in image_files:
            os.remove(image_file)  # delete images
        print("folder cleaned")
    else:
        print("folder empty")


if __name__ == '__main__':
    ImageDir = 'data/SubT_MRS_t1/image_left'
    timestampsDir = 'data/SubT_MRS_t1/image_left/timestamps.txt'
    gdthDir = 'data/SubT_MRS_t1/ground_truth_path.csv'
    target = 'data/targetImageFolder'
    list1 = returnPt(timestampsDir, gdthDir)
    clear_images_in_folder(target)
    copy_selected_images(list1, ImageDir, target)
    print(len(list1))
