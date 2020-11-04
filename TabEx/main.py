import tabx
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PIL import Image
import pytesseract
img = cv2.imread('<path to image>',0)

print(tabx.main(img))
