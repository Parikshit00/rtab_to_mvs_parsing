import csv
import sys
import pandas as pd
import numpy as np
import os
from os import path
import shutil
import argparse
import natsort




class mvsparser:
    def __init__(self):
        self._parser = parse()
    
    
    def parse(txt_file, intrinsics, input_path, output_path):
        # Pandas dataframe
        df = pd.read_csv(txt_file, delim_whitespace = True, header = None)
        # Convert to array for inverse
        poses = df.values
        # Augmentation matrix for producing square matrix
        aug = np.array([0.0, 0.0, 0.0, 1.0])
        final_inversed = None
        for array in poses:
            pose_augmented = np.append(array, aug).reshape(4,4)
            pose_inversed = np.linalg.inv(pose_augmented)
            pose_inversed = pose_inversed[:-1].flatten()
            if final_inversed is None:
                final_inversed = pose_inversed
            else:
                final_inversed = np.vstack((final_inversed, pose_inversed))
        # Easier to swap the columns by converting into dataframe
        # The pose.txt file contains the extrinsic values in the specified format
        df = pd.DataFrame(data=final_inversed, columns=['R11', 'R12', 'R13', 'Tx', 'R21', 'R22', 'R23', 'Ty', 'R31', 'R32', 'R33', 'Tz'])
        # Converting the extrinsic matrix into Texrecon accepted format
        df = df[['Tx', 'Ty', 'Tz', 'R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33']]
        # Converting back to array format
        final_poses = df.values
        # Adding intrinsic matrix to pair with extrinsic matrix of every imaage
        file_list = os.listdir(input_path + "/images/rgb/")
        file_list = natsort.natsorted(file_list)
        for filename, row in zip(file_list, final_poses):
            print ("Creating .cam file for ", filename)
            with open(os.path.join(output_path, os.path.splitext(filename)[0]) + ".cam","w") as f:
                #print(os.path.join(output_path, os.path.splitext(filename)[0]) + ".cam","w")
                f.write("\n".join(" ".join(map(str, x)) for x in (row, intrinsics)))



def main(input_path, output_path):
    # Specify the poses.txt filename. Must be in the same folder as this script
    txt_file = input_path + '/poses/camera.txt'
    # The intrinsic matrix
    intrinsics = np.array([602.0198974609375/1280, 0.0, 0.0, 602.0198974609375/601.69488525390625, 637.13360595703125/1280, 365.44882202148438/720])
    #Create the output folder and delete previous if already exists
    output_folder = output_path+"/cam"
    if(path.exists(output_folder)):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    #Parsing .txt file to .cam files
    parse(txt_file, intrinsics, input_path, output_folder)
    
    #copying the images into the cam folder 
    images_folder = input_path + "/images/rgb/"
    for imageFile in os.listdir(images_folder):
        print("Copying the file" +os.path.join(images_folder,imageFile))
        shutil.copy(os.path.join(images_folder,imageFile),output_folder)



# calling the main() function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RTABMap-To-MVS Preprocessing. Transforms poses text files from RTAB map to cam files. Make sure that the images and poses are stored inside the same directory. The images and poses should be stored as rootdir/images/rgb/ and rootdir/poses/camera.txt respectively.")
    parser.add_argument("--source", help="Path to root directory of rtabmap export images and poses text file")
    parser.add_argument("--output", help="Path to output location for cam folder")
    args = parser.parse_args()
    input_path = args.source
    output_path = args.output

    if not args.source or not args.output:
        parser.print_help(sys.stderr)
        sys.exit(1)

    main(input_path, output_path)
