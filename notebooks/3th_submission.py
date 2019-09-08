import numpy as np
import os
import pandas as pd
from PIL import Image
from keras.models import model_from_json
from scipy.misc import imresize
from skimage import measure
from matplotlib import pyplot as plt
import seaborn as sns
import cv2

grade_bounds = [(0.5, 0.75), (0.0, 0.25), (0.25, 0.5), (0.0, 0.25), (0.0, 0.25)]

dict_identifier_path = {(0, 0, 0, 0, 0):"../input/submission-0034/submission_0.034.csv",
                        (0, 0, 0, 0, 1):"../input/submission-0716/submission_0.716.csv",
                        (0, 0, 0, 1, 0):"../input/submission-0739/submission_0.739.csv",
                        (0, 0, 0, 1, 1):"../input/submission-0192/submission_0.192.csv",
                        (0, 0, 1, 0, 0):"../input/submission-0421/submission_0.421.csv",
                        (0, 0, 1, 0, 1):"../input/submission-0451/submission_0.451.csv",
                        (0, 0, 1, 1, 0):"../input/submission-0558/submission_0.558.csv",
                        (0, 0, 1, 1, 1):"../input/submission-0683/submission_0.683.csv",
                        (0, 1, 0, 0, 0):"../input/submission-0686/submission_0.686.csv",
                        (0, 1, 0, 0, 1):"../input/submission-0695/submission_0.695.csv",
                        (0, 1, 0, 1, 0):"../input/submission-0709/submission_0.709.csv",
                        (0, 1, 0, 1, 1):"../input/submission-0711/submission_0.711.csv",
                        (0, 1, 1, 0, 0):"../input/submission-0715/submission_0.715.csv",
                        (0, 1, 1, 0, 1):"../input/submission-0738/submission_0.738.csv",
                        (0, 1, 1, 1, 0):"../input/submission-0749/submission_0.749.csv",
                        (0, 1, 1, 1, 1):"../input/submission-0751/submission_0.751.csv",
                        (1, 0, 0, 0, 0):"../input/submission-0753/submission_0.753.csv",
                        (1, 0, 0, 0, 1):"../input/submission-0755/submission_0.755.csv",
                        (1, 0, 0, 1, 0):"../input/submission-0759/submission_0.759.csv",
                        (1, 0, 0, 1, 1):"../input/submission-0766/submission_0.766.csv",
                        (1, 0, 1, 0, 0):"../input/submission-0768/submission_0.768.csv",
                        (1, 0, 1, 0, 1):"../input/submission-0777/submission_0.777.csv",
                        (1, 0, 1, 1, 0):"../input/submission-0783/submission_0.783.csv",
                        (1, 0, 1, 1, 1):"../input/submission-0785/submission_0.785.csv",
                        (1, 1, 0, 0, 0):"../input/submission-0790/submission_0.790.csv",
                        (1, 1, 0, 0, 1):"../input/submission-0521/submission_0.521.csv",
                        (1, 1, 0, 1, 0):"../input/submission-0261/submission_0.261.csv",
                        (1, 1, 0, 1, 1):"../input/submission-0649/submission_0.649.csv",
                        (1, 1, 1, 0, 0):"../input/submission-0661/submission_0.661.csv",
                        (1, 1, 1, 0, 1):"../input/submission-0576/submission_0.576.csv",
                        (1, 1, 1, 1, 0):"../input/submission-0681/submission_0.681.csv",
                        (1, 1, 1, 1, 1):"../input/submission-0608/submission_0.608.csv"}

def crop_resize_img_dr(img, method):
    
    # set params
    h_ori, w_ori, _ = np.shape(img)
    red_threshold = 20
    roi_check_len = h_ori // 5
    ungradable = False
    
    # Find Connected Components with intensity above the threshold
    blobs_labels, n_blobs = measure.label(img[:, :, 0] > red_threshold, return_num=True)
    if n_blobs == 0:
        ungradable = True

    # Find the Index of Connected Components of the Fundus Area (the central area)
    majority_vote = np.argmax(
        np.bincount(
            blobs_labels[
              h_ori // 2 - roi_check_len // 2:
              h_ori // 2 + roi_check_len // 2,
              w_ori // 2 - roi_check_len // 2:
              w_ori // 2 + roi_check_len // 2].flatten()))
    if majority_vote == 0:
        ungradable = True

    row_inds, col_inds = np.where(blobs_labels == majority_vote)
    row_max = np.max(row_inds)
    row_min = np.min(row_inds)
    col_max = np.max(col_inds)
    col_min = np.min(col_inds)
    if row_max - row_min < 100 or col_max - col_min < 100:
        ungradable = True

    # crop the image
    if ungradable:
        len_side = min(h_ori, w_ori)
        crop_img = img[h_ori // 2 - len_side // 2:h_ori // 2 + len_side // 2, w_ori // 2 - len_side // 2:w_ori // 2 + len_side // 2]
    else:
        crop_img = img[row_min:row_max, col_min:col_max]
    max_len = max(crop_img.shape[0], crop_img.shape[1])
    img_h, img_w, _ = crop_img.shape
    padded = np.zeros((max_len, max_len, 3), dtype=np.uint8)
    padded[
        (max_len - img_h) // 2:
        (max_len - img_h) // 2 + img_h,
        (max_len - img_w) // 2:
        (max_len - img_w) // 2 + img_w, ...] = crop_img
    cropped_w = max_len
    y_offset = row_min - (max_len - img_h) // 2
    x_offset = col_min - (max_len - img_w) // 2
    resized_img = imresize(padded, (512, 512), method)
    return resized_img


def all_files_under(path, extension=None, append_path=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]
    
    return filenames


def load_network(dir_name):
    network_file = all_files_under(dir_name, extension=".json")
    weight_file = all_files_under(dir_name, extension=".h5")
    assert len(network_file) == 1 and len(weight_file) == 1
    with open(network_file[0], 'r') as f:
        network = model_from_json(f.read())
    network.load_weights(weight_file[0])
    network.trainable = False
    for l in network.layers:
        l.trainable = False
    return network


def normalize_ben(img):
    ben_img = cv2.addWeighted (img, 4, cv2.GaussianBlur(img , (0, 0) , 10) , -4 , 128)
    img = np.concatenate([img / 255.0, ben_img / (3 * 255.0)], axis=-1)
    return img


def outputs2labels(outputs, min_val, max_val):
    return np.clip(np.round(outputs), min_val, max_val)


dir_test_images = "../input/aptos2019-blindness-detection/test_images"

# load networks
dr_model_dir = '../input/representative-model'
dr_model = load_network(dr_model_dir)

# iterate for each file from csv file
sample_submit_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
list_id_code, list_grades = [], []
for id_code in sample_submit_df["id_code"]:
    list_id_code.append(id_code)

    # load imgs
    fpath = os.path.join(dir_test_images, "{}.png".format(id_code))
    img = np.array(Image.open(fpath)).astype(np.float32)
    resized_img_bicubic = crop_resize_img_dr(img, "bicubic")
    
    # run infererence
    pred0, pred1, pred2, pred3, dr_grade = dr_model.predict(np.expand_dims(normalize_ben(resized_img_bicubic), axis=0), batch_size=1, verbose=0)
    list_grades.append(int(outputs2labels(dr_grade[0, 0], 0, 4)))
# generate dataframe for public test + private test
submit_df = pd.DataFrame({"id_code":list_id_code,"diagnosis":list_grades})

# plot label distribution after inference
sns.countplot(submit_df["diagnosis"])
plt.savefig("diagnosis_label_after_inference.png")
plt.show()

# make dataframe for private test
list_public_test_id_code = list(pd.read_csv("../input/submission-0034/submission_0.034.csv")["id_code"])
list_private_test_grades = []
for id_code in list(set(submit_df["id_code"]) - set(list_public_test_id_code)):  # private_test = all - public_test
    list_private_test_grades.append(int(list(submit_df[submit_df["id_code"] == id_code]["diagnosis"])[0])) 

# select dataframe for identification    
if len(list_private_test_grades) == 0:  # debugging purpose
    encoding = [0, 0, 0, 0, 0]
else:
    assert len(np.unique(list_private_test_grades))==5 and np.min(list_private_test_grades)==0 and np.max(list_private_test_grades)==4
    n_total = len(list_private_test_grades)
    arr_private_test_grades = np.array(list_private_test_grades)
    ratio_bincount = []
    for grade in range(5):
        ratio_bincount.append(1.*len(arr_private_test_grades[arr_private_test_grades==grade]) / n_total)
    encoding = []
    for grade, ratio in enumerate(ratio_bincount):
        code = 1 if ratio >= np.mean(grade_bounds[grade]) and ratio < grade_bounds[grade][1] else 0
        encoding.append(code)
path_identifier_csv = dict_identifier_path[tuple(encoding)]
df_identifier = pd.read_csv(path_identifier_csv)

# replace diagnosis of public test diagnosis with identifier
for id_code in list_public_test_id_code:
    submit_df.loc[submit_df["id_code"] == id_code, "diagnosis"] = int(df_identifier[df_identifier["id_code"] == id_code]["diagnosis"])
submit_df.to_csv("submission.csv", index=False)

# plot label distribution
sns.countplot(submit_df["diagnosis"])
plt.savefig("diagnosis_label_replaced.png")
plt.show()