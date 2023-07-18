import json
from typing import List
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os, sys, glob
from pathlib import Path, PurePath
import yaml
import shutil
from tqdm import tqdm
import csv
import argparse

from labelme2coco import labelme2coco
from dataset import Dataset


class labelme2yolo:
    def __init__(self, dataset=None):
        print("ready to import coco data")
        self.dataset = dataset
        self.schema = [
            "img_folder",
            "img_filename",
            "img_path",
            "img_id",
            "img_width",
            "img_height",
            "img_depth",
            "ann_segmented",
            "ann_bbox_xmin",
            "ann_bbox_ymin",
            "ann_bbox_xmax",
            "ann_bbox_ymax",
            "ann_bbox_width",
            "ann_bbox_height",
            "ann_area",
            "ann_segmentation",
            "ann_iscrowd",
            "ann_pose",
            "ann_truncated",
            "ann_difficult",
            "cat_id",
            "cat_name",
            "cat_supercategory",
            "split",
            "annotated",
        ]

    def _ReindexCatIds(self, df, cat_id_index=0):
        """
        Reindex the values of the cat_id column so that that they start from an int (usually 0 or 1) and
        then increment the cat_ids to index + number of categories.
        It's useful if the cat_ids are not continuous, especially for dataset subsets,
        or combined multiple datasets. Some models like Yolo require starting from 0 and others
        like Detectron require starting from 1.
        """
        assert isinstance(cat_id_index, int), "cat_id_index must be an int."

        # Convert empty strings to NaN and drop rows with NaN cat_id
        df_copy = df.replace(r"^\s*$", np.nan, regex=True)
        df_copy = df_copy[df.cat_id.notnull()]
        # Coerce cat_id to int
        df_copy["cat_id"] = pd.to_numeric(df_copy["cat_id"])

        # Map cat_ids to the range [cat_id_index, cat_id_index + num_cats)
        unique_ids = np.sort(df_copy["cat_id"].unique())
        ids_dict = dict((v, k) for k, v in enumerate(unique_ids, start=cat_id_index))
        df_copy["cat_id"] = df_copy["cat_id"].map(ids_dict)

        # Write back to the original dataframe
        df["cat_id"] = df_copy["cat_id"]

    def ImportCoco(self, path, path_to_images=None, name=None, encoding="utf-8"):
        """
        This function takes the path to a JSON file in COCO format as input. It returns a PyLabel dataset object that contains the annotations.

        Returns:
            PyLabel dataset object.

        Args:
            path (str):The path to the JSON file with the COCO annotations.
            path_to_images (str): The path to the images relative to the json file.
                If the images are in the same directory as the JSON file then omit this parameter.
                If the images are in a different directory on the same level as the annotations then you would
                set `path_to_images='../images/'`
            name (str): This will set the dataset.name property for this dataset.
                If not specified, the filename (without extension) of the COCO annotation file file will be used as the dataset name.
            encoding (str): Default is 'utf-8. Encoding of the annotations file(s).
        Example:
            >>> from pylabel import importer
            >>> dataset = importer.ImportCoco("coco_annotations.json")
        """
        parent_path = os.path.dirname(path[0])
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        add_path = None
        image_dir = []
        
        for file_path in glob.glob(os.path.join(parent_path, '*')):
            if os.path.isdir(file_path):
                for sub_path in glob.glob(os.path.join(file_path, '*')):
                    if os.path.isfile(sub_path) and any(sub_path.lower().endswith(ext) for ext in image_extensions):
                        image_dir.append(file_path)
                        break        
        
        for path in path:        
            with open(path, encoding=encoding) as cocojson:
                annotations_json = json.load(cocojson)

            if 'val' in path:
                add_path = "/labels/val"
            elif 'train' in path:
                add_path = "/labels/train"
            else:
                add_path = "/labels"
    
            # Store the 3 sections of the json as seperate json arrays
            images = pd.json_normalize(annotations_json["images"])
            images.columns = "img_" + images.columns
            try:
                images["img_folder"]
            except:
                images["img_folder"] = ""
            # print(images)
    
            # If the user has specified a different image folder then use that one
            if path_to_images != None:
                images["img_folder"] = path_to_images
    
            astype_dict = {"img_width": "int64", "img_height": "int64", "img_depth": "int64"}
            astype_keys = list(astype_dict.keys())
            for element in astype_keys:
                if element not in images.columns:
                    astype_dict.pop(element)
            # print(astype_dict)
            # images = images.astype({'img_width': 'int64','img_height': 'int64','img_depth': 'int64'})
            images = images.astype(astype_dict)
    
            annotations = pd.json_normalize(annotations_json["annotations"])
            annotations.columns = "ann_" + annotations.columns
    
            categories = pd.json_normalize(annotations_json["categories"])
            categories.columns = "cat_" + categories.columns
    
            # Converting this to string resolves issue #23
            categories.cat_id = categories.cat_id.astype(str)
    
            df = annotations
    
            # Converting this to string resolves issue #23
            df.ann_category_id = df.ann_category_id.astype(str)
    
            df[["ann_bbox_xmin", "ann_bbox_ymin", "ann_bbox_width", "ann_bbox_height"]] = pd.DataFrame(
                df.ann_bbox.tolist(), index=df.index
            )
            df.insert(8, "ann_bbox_xmax", df["ann_bbox_xmin"] + df["ann_bbox_width"])
            df.insert(10, "ann_bbox_ymax", df["ann_bbox_ymin"] + df["ann_bbox_height"])
    
            # debug print(df.info())
    
            # Join the annotions with the information about the image to add the image columns to the dataframe
            df = pd.merge(images, df, left_on="img_id", right_on="ann_image_id", how="left")
            df = pd.merge(df, categories, left_on="ann_category_id", right_on="cat_id", how="left")
    
            # Rename columns if needed from the coco column name to the pylabel column name
            df.rename(columns={"img_file_name": "img_filename"}, inplace=True)
            df.rename(columns={"img_path": "img_path"}, inplace=True)
            df["img_path"] = add_path
            # Drop columns that are not in the schema
            df = df[df.columns.intersection(self.schema)]
    
            # Add missing columns that are in the schema but not part of the table
            df[list(set(self.schema) - set(df.columns))] = ""
    
            # Reorder columns
            df = df[self.schema]
            df.index.name = "id"
            df.annotated = 1
    
            # Fill na values with empty strings which resolved some errors when
            # working with images that don't have any annotations
            df.fillna("", inplace=True)
    
            # These should be strings
            df.cat_id = df.cat_id.astype(str)
    
            # These should be integers
            df.img_width = df.img_width.astype(int)
            df.img_height = df.img_height.astype(int)
            
            dataset = Dataset(df)
    
            # Assign the filename (without extension) as the name of the dataset
            if name == None:
                dataset.name = Path(path).stem
            else:
                dataset.name = name
    
            dataset.path_to_annotations = PurePath(path).parent
            
            if self.dataset is not None:
                # Append the new dataset to the existing dataset
                self.dataset.df = pd.concat([self.dataset.df, dataset.df], ignore_index=True)
            else:
                self.dataset = dataset
                                
    def ExportToYoloV5(
        self,
        output_path="training/labels",
        yaml_file="dataset.yaml",
        copy_images=False,
        use_splits=False,
        cat_id_index=None,
        segmentation=False,
    ):
        """Writes annotation files to disk in YOLOv5 format and returns the paths to files.

        Args:

            output_path (str):
                This is where the annotation files will be written.
                If not-specified then the path will be derived from the .path_to_annotations and
                .name properties of the dataset object. If you are exporting images to train a model, the recommended path
                to use is 'training/labels'.
            yaml_file (str):
                If a file name (string) is provided, a YOLOv5 YAML file will be created with entries for the files
                and classes in this dataset. It will be created in the parent of the output_path directory.
                The recommended name for the YAML file is 'dataset.yaml'.
            copy_images (boolean):
                If True, then the annotated images will be copied to a directory next to the labels directory into
                a directory named 'images'. This will prepare your labels and images to be used as inputs to
                train a YOLOv5 model.
            use_splits (boolean):
                If True, then the images and annotations will be moved into directories based on the values in the split column.
                For example, if a row has the value split = "train" then the annotations for that row will be moved to directory
                /train. If a YAML file is specificied then the YAML file will use the splits to specify the folders user for the
                train, val, and test datasets.
            cat_id_index (int):
                Reindex the cat_id values so that that they start from an int (usually 0 or 1) and
                then increment the cat_ids to index + number of categories continuously.
                It's useful if the cat_ids are not continuous in the original dataset.
                Yolo requires the set of annotations to start at 0 when training a model.
            segmentation (boolean):
                If true, then segmentation annotations will be exported instead of bounding box annotations.
                If there are no segmentation annotations, then no annotations will be empty.

        Returns:
            A list with 1 or more paths (strings) to annotations files. If a YAML file is created
            then the first item in the list will be the path to the YAML file.

        Examples:
            >>> dataset.export.ExportToYoloV5(output_path='training/labels',
            >>>     yaml_file='dataset.yaml', cat_id_index=0)
            ['training/dataset.yaml', 'training/labels/frame_0002.txt', ...]

        """
        ds = self.dataset        

        # Inspired by https://github.com/aws-samples/groundtruth-object-detection/blob/master/create_annot.py
        yolo_dataset = ds.df.copy(deep=True)
        # Convert nan values in the split column from nan to '' because those are easier to work with with when building paths
        yolo_dataset.split = yolo_dataset.split.fillna("")
        print(yolo_dataset.split)
        # Create all of the paths that will be used to manage the files in this dataset
        path_dict = {}

        # The output path is the main path that will be used to create the other relative paths
        path = PurePath(output_path)
        path_dict["label_path"] = output_path
        # The /images directory should be next to the /labels directory
        path_dict["image_path"] = str(PurePath(path.parent, "images"))
        # The root directory is in parent of the /labels and /images directories
        path_dict["root_path"] = str(PurePath(path.parent))
        # The YAML file should be in root directory
        path_dict["yaml_path"] = str(PurePath(path_dict["root_path"], yaml_file))
        # The root directory will usually be next to the yolov5 directory.
        # Specify the relative path
        path_dict["root_path_from_yolo_dir"] = str(PurePath("../"))
        # If these default values to not match the users environment then they can manually edit the YAML file

        if copy_images:
            # Create the folder that the images will be copied to
            Path(path_dict["image_path"]).mkdir(parents=True, exist_ok=True)

        # Drop rows that are not annotated
        # Note, having zero annotates can still be considered annotated
        # in cases when are no objects in the image thats should be indentified
        yolo_dataset = yolo_dataset.loc[yolo_dataset["annotated"] == 1]

        # yolo_dataset["cat_id"] = (
        #     yolo_dataset["cat_id"].astype("float").astype(pd.Int32Dtype())
        # )

        yolo_dataset.cat_id = yolo_dataset.cat_id.replace(r"^\s*$", np.nan, regex=True)

        pd.to_numeric(yolo_dataset["cat_id"])

        if cat_id_index != None:
            assert isinstance(cat_id_index, int), "cat_id_index must be an int."
            self._ReindexCatIds(yolo_dataset, cat_id_index)

        # Convert empty bbox coordinates to nan to avoid math errors
        # If an image has no annotations then an empty label file will be created
        yolo_dataset.ann_bbox_xmin = yolo_dataset.ann_bbox_xmin.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_ymin = yolo_dataset.ann_bbox_ymin.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_width = yolo_dataset.ann_bbox_width.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_height = yolo_dataset.ann_bbox_height.replace(
            r"^\s*$", np.nan, regex=True
        )

        # If segmentation = False then export bounding boxes
        if segmentation == False:
            yolo_dataset["center_x_scaled"] = (
                yolo_dataset["ann_bbox_xmin"] + (yolo_dataset["ann_bbox_width"] * 0.5)
            ) / yolo_dataset["img_width"]
            yolo_dataset["center_y_scaled"] = (
                yolo_dataset["ann_bbox_ymin"] + (yolo_dataset["ann_bbox_height"] * 0.5)
            ) / yolo_dataset["img_height"]
            yolo_dataset["width_scaled"] = (
                yolo_dataset["ann_bbox_width"] / yolo_dataset["img_width"]
            )
            yolo_dataset["height_scaled"] = (
                yolo_dataset["ann_bbox_height"] / yolo_dataset["img_height"]
            )

        # Create folders to store annotations
        if output_path == None:
            dest_folder = PurePath(ds.path_to_annotations, yolo_dataset.iloc[0].img_folder)
        else:
            dest_folder = output_path

        os.makedirs(dest_folder, exist_ok=True)

        unique_images = yolo_dataset["img_filename"].unique()
        output_file_paths = []
        pbar = tqdm(desc="Exporting YOLO files...", total=len(unique_images))
        for img_filename in unique_images:
            df_single_img_annots = yolo_dataset.loc[yolo_dataset.img_filename == img_filename]

            basename, _ = os.path.splitext(img_filename)
            annot_txt_file = basename + ".txt"
            # Use the value of the split collumn to create a directory
            # The values should be train, val, test or ''
            if use_splits:
                split_dir = df_single_img_annots.iloc[0].split
            else:
                split_dir = ""
            destination = str(PurePath(dest_folder, split_dir, annot_txt_file))
            Path(
                dest_folder,
                split_dir,
            ).mkdir(parents=True, exist_ok=True)

            # If segmentation = false then output bounding boxes
            if segmentation == False:
                df_single_img_annots.to_csv(
                    destination,
                    index=False,
                    header=False,
                    sep=" ",
                    float_format="%.4f",
                    columns=[
                        "cat_id",
                        "center_x_scaled",
                        "center_y_scaled",
                        "width_scaled",
                        "height_scaled",
                    ],
                )

            # If segmentation = true then output the segmentation mask
            else:
                # Create one file for image
                with open(destination, "w") as file:
                    # Create one row per row in the data frame
                    for i in range(0, df_single_img_annots.shape[0]):
                        row = str(df_single_img_annots.iloc[i].cat_id)
                        segmentation_array = df_single_img_annots.iloc[i].ann_segmentation[0]

                        # Iterate through every value of the segmentation array
                        # To normalize the coordinates from 0-1
                        for index, l in enumerate(segmentation_array):
                            # The first number in the array is the x value so divide by the width
                            if index % 2 == 0:
                                row += " " + (
                                    str(
                                        segmentation_array[index]
                                        / df_single_img_annots.iloc[i].img_width
                                    )
                                )
                            else:
                                # The first number in the array is the x value so divide by the height
                                row += " " + (
                                    str(
                                        segmentation_array[index]
                                        / df_single_img_annots.iloc[i].img_height
                                    )
                                )

                        file.write(row + "\n")

            output_file_paths.append(destination)

            if copy_images:
                source_image_path = str(
                    Path(
                        df_single_img_annots.iloc[0].img_folder,
                        df_single_img_annots.iloc[0].img_filename,
                    )
                )

                current_file = Path(source_image_path)
                assert (
                    current_file.is_file
                ), f"File does not exist: {source_image_path}. Check img_folder column values."
                Path(path_dict["image_path"], split_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy(
                    str(source_image_path),
                    str(PurePath(path_dict["image_path"], split_dir, img_filename)),
                )
            pbar.update()

        # Create YAML file
        if yaml_file:
            # Make a set with all of the different values of the split column
            splits = set(yolo_dataset.split)
            # Build a dict with all of the values that will go into the YAML file
            dict_file = {}
            dict_file["path"] = path_dict["root_path_from_yolo_dir"]
            

            # If train is one of the splits, append train to path
            if use_splits and "train" in splits:
                dict_file["train"] = str(PurePath(path_dict["image_path"], "train"))
            else:
                dict_file["train"] = path_dict["image_path"]

            # If val is one of the splits, append val to path
            if use_splits and "val" in splits:
                dict_file["val"] = str(PurePath(path_dict["image_path"], "val"))
            else:
                # If there is no val split, use the train split as the val split
                dict_file["val"] = dict_file["train"]

            # If test is one of the splits, make a test param and add test to the path
            if use_splits and "test" in splits:
                dict_file["test"] = str(PurePath(path_dict["image_path"], "test"))

            dict_file["nc"] = ds.analyze.num_classes
            dict_file["names"] = ds.analyze.classes

            # Save the yamlfile
            with open(path_dict["yaml_path"], "w") as file:
                documents = yaml.dump(dict_file, file, encoding="utf-8", allow_unicode=True)                
                output_file_paths = [path_dict["yaml_path"]] + output_file_paths

        return output_file_paths


if __name__ == "__main__":
    labelme2coco = labelme2coco()
    labelme2yolo = labelme2yolo()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, help="Please input the path of the labelme json files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/labelme2coco/",
        help="Please set the path for the ouput directory.",
    )
    parser.add_argument(
        "--train_rate",
        type=float,
        nargs="?",
        default=0,
        help="Please input the validation dataset size, for example 0.1 ",
    )

    args = parser.parse_args(sys.argv[1:])
    if args.input_dir is not None:
        input = args.input_dir
        output = args.output_dir
        rate = args.train_rate

        coco = labelme2coco.convert(
            labelme_folder=input,
            export_dir=output,
            train_split_rate=rate,
        )

        json_path = []
        for file_path in glob.glob(os.path.join(output, '*')):
            if os.path.isfile(file_path) and file_path.endswith('.json'):
                json_path.append(file_path)
                
        labelme2yolo.ImportCoco(path=json_path, path_to_images="", name="data_coco")
        labelme2yolo.ExportToYoloV5(output_path=parent_path+add_path, copy_images=True, segmentation=True)[1]
    else:
        print("Please define the path for labelme dataset location")
