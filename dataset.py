"""The dataset is the primary object that you will interactive with when using PyLabel.
All other modules are sub-modules of the dataset object. 
"""

from analyze import Analyze
from splitter import Split

import numpy as np


class Dataset:
    def __init__(self, df):
        self.df = df
        """Pandas Dataframe: The dataframe where annotations are stored. This dataframe can be read directly
        to query the contents of the dataset. You can also edit this dataframe to filter records or edit the 
        annotations directly. 

        Example: 
            >>> dataset.df
        """
        self.name = "dataset"
        """string: Default is 'dataset'. A friendly name for your dataset that is used as part of the filename(s)
        when exporting annotation files. 
        """
        self.path_to_annotations = ""
        """string: Default is ''. The path to the annotation files associated with the dataset. When importing, 
        this will be path to the directory where the annotations are stored.  By default, annotations will be exported
        to the same directory. Changing this value will change where the annotations are exported to.  
        """
        self.analyze = Analyze(dataset=self)
        """See pylabel.analyze module."""
        self.splitter = Split(dataset=self)
        """See pylabel.splitter module."""
