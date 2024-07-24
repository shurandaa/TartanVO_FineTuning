import pandas as pd
import shutil
import os
import glob


def returnPt(image_csv, ground_truth):
    cul = 0
    # Read motion data
    motion_data = pd.read_csv(ground_truth)
    motion_timestamps = motion_data['#timestamp'].tolist()  # Convert timestamps to a list
    # Read image timestamps
    image_data = pd.read_csv(image_csv)
    image_timestamps = image_data['#timestamp [ns]'].tolist() #Read each row's timestamp as an element of the list
    selected_indices = []  # Store the indices of the selected image timestamps
    selected_groundtruth = [] #Store the corresponding ground truth

    # Iterate over each pair of consecutive motion data timestamps
    for i in range(len(motion_timestamps) - 1):
        start = motion_timestamps[i]
        end = motion_timestamps[i + 1]

        #Find image timestamps between the two motion timestamps
        for index, timestamp in enumerate(image_timestamps):
            if start <= int(timestamp) < end:
                selected_indices.append(index)
                selected_groundtruth.append(i)
                break  # Stop after finding the first one. If you need to find all images between the two timestamps, remove this break

    for i in range(len(selected_indices)):
        cul += 1
    # Output the results
    print(cul)
    return selected_indices, selected_groundtruth


def copy_selected_images(numbers_list, source_dir, target_dir, image_data):
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
        file_name = image_data.iloc[number, 1]
        source_file = os.path.join(source_dir, f"{file_name}")
        target_file = os.path.join(target_dir, f"{file_name}")

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
        print("no Images in the folder")


def copyGroundtruth(number_list, target_groundtruth, groundtruth):
    if os.path.exists(target_groundtruth) and os.path.isdir(target_groundtruth):
        shutil.rmtree(target_groundtruth)
        print(f"Directory '{target_groundtruth}' has been removed.")
    data = pd.read_csv(groundtruth)
    data_selected = data.iloc[number_list]
    data_selected.to_csv(target_groundtruth, index=False)


if __name__ == '__main__':
    ImageDir = 'data/eURoc_DataMH4/data'
    timestampsDir = 'data/eURoc_DataMH4/data/data.csv'
    gdthDir = 'data/eURoc_DataMH4/data.csv'
    target = 'data/targetImageFolder'
    image_data = pd.read_csv('data/eURoc_DataMH4/data/data.csv')
    targetgroundtruth = 'data/eURoc_DataMH4/groundtruthSelected.csv'
    list1, list2 = returnPt(timestampsDir, gdthDir)
    clear_images_in_folder(target)
    copy_selected_images(list1, ImageDir, target, image_data)
    copyGroundtruth(list2, targetgroundtruth, gdthDir)

    print(len(list1))