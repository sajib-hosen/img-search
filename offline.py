from PIL import Image
from pathlib import Path
import numpy as np

from feature_extractor import FeatureExtractor

if __name__ == "__main__":
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        feature = fe.extract(img=Image.open(img_path)) # creating image feature
        feature_path = Path("./static/feature")/(img_path.stem + ".npy") # creating feature path
        np.save(feature_path, feature) # save the features in the feature folder

        # print(img_path)
        # print(feature)
        # print(feature_path)

        # https://www.youtube.com/watch?v=M0Y9_vBmYXU&t=730s
        # https://github.com/matsui528/sis
        # https://www.simple-image-search.xyz/