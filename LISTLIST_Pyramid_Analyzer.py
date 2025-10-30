"""
Merged Pyramid Analysis Code
Combines all modules for pyramid detection and height measurement from SEM images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage, signal
from skimage import measure, color, io
import scipy.interpolate as interpolate
from PIL import Image
import pandas as pd
import os
import json
from typing import List, Tuple, Union, Optional, Dict, Any, Callable
import bottleneck as bn
import tkinter as tk
from tkinter.filedialog import askopenfilename

# ============================================================================
# T1CV Module - Computer Vision Utilities
# ============================================================================

def dirs_to_point(point: np.ndarray, points: np.ndarray):
    return points - point

def dirs_to_line_orth(startpoint: np.ndarray, direction_vector: np.ndarray, points: np.ndarray):
    line_direction = direction_vector / np.linalg.norm(direction_vector)
    orth_direction = np.array([line_direction[1], -line_direction[0]])
    orth_direction /= np.linalg.norm(orth_direction)
    return np.dot(points - startpoint, orth_direction).reshape(-1, 1) * orth_direction

def scalars_to_line_orth(startpoint: np.ndarray, direction_vector: np.ndarray, points: np.ndarray):
    line_direction = direction_vector / np.linalg.norm(direction_vector)
    orth_direction = np.array([line_direction[1], -line_direction[0]])
    orth_direction /= np.linalg.norm(orth_direction)
    return np.dot(points - startpoint, orth_direction).reshape(-1, 1)

def dirs_to_point_with_line(startpoint: np.ndarray, direction_vector: np.ndarray, points: np.ndarray):
    line_direction = direction_vector / np.linalg.norm(direction_vector)
    return np.dot(points - startpoint, line_direction) * line_direction

def scalars_to_with_line(startpoint: np.ndarray, direction_vector: np.ndarray, points: np.ndarray):
    line_direction = direction_vector / np.linalg.norm(direction_vector)
    return (np.dot(points - startpoint, line_direction)).reshape(-1, 1)

def arrow(canvas, startpoint, endpoint, color = (0, 0, 255), thickness = 2):
    startpoint = np.array(startpoint) 
    endpoint = np.array(endpoint)
    cv2.arrowedLine(canvas, tuple(startpoint[::-1].astype(np.int32)), tuple(endpoint[::-1].astype(np.int32)), color, thickness)

def line(canvas, startpoint, endpoint, color = (0, 0, 255), thickness = 2):
    startpoint = np.array(startpoint) 
    endpoint = np.array(endpoint)
    cv2.line(canvas, tuple(startpoint[::-1].astype(np.int32)), tuple(endpoint[::-1].astype(np.int32)), color, thickness)

def show(img, s="img"):
    if isinstance(img, list):
        for i in img:
            show(i, s + str(i))
    else:
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
            plt.plot()
        cv2.imshow(s, img)
        cv2.waitKey(0)

def get_regions_and_add_function():
    regions = {}
    def add_region(region):
        s = np.sum(region)
        print(s)
        s = int(s)
        if s in regions:
            regions[s].append(region)
        else:
            regions[s] = [region]
    return regions, add_region

def get_contour_masks(thresh, level = None):
    regions, add_region = get_regions_and_add_function()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    parsed, top = parse_hierarchy(contours, hierarchy, level = level)
    for i, contour in enumerate(contours):
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        parsed[i].region_mask = mask
        region = cv2.bitwise_and(thresh, mask)
        add_region(mask)
    return regions, parsed, top

def remove_bottom_bar(img):
    return img[:-50, :]

def partition_contours(region_dict, minsum = 1200, maxsum = None):
    high = []
    low = []
    mid = []
    for i in region_dict.keys():
        if maxsum is not None and i > maxsum:
            high += region_dict[i]    
        elif minsum is not None and i < minsum:
            low += region_dict[i]
        else:
            mid += region_dict[i]
    return mid, low, high

class Contour():
    def __init__(self, contour, parent = None, level = None):
        self.contour = contour
        self.children = []
        self.parent = parent
        self.region_mask = None
        self.level = level

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

import sys
sys.setrecursionlimit(2000)
    
def parse_hierarchy(contours, hierarchy, minimum_sum = 1000, level = None):
    """
    c: List[np.ndarray]; list of contours  
    h: np.ndarray; hierarchy of contours; h.shape == (1, len(c), 4)
    """
    print("hierarchy.shape", hierarchy.shape)
    checked = np.zeros(len(hierarchy[0, :, 0]) + 1)
    checked[-1] = 1
    C = [None] * len(hierarchy[0, :, 0]) + [None]
    i = 0
    top = set()
    while not np.all(checked):
        def check(i):
            print("checking", i, checked.shape)
            if checked[i]:
                return
            h = hierarchy[0, i, :]
            c = contours[i]
            sib_next, sib_prev, child_first, parent = h[0], h[1], h[2], h[3]
            if not checked[parent]:
                check(parent)
            checked[i] = 1
            C[i] = Contour(c, C[parent])
            if parent != -1:
                C[parent].add_child(C[i])
            else:
                top.add(C[i])
            checked[i] = 1
            if not checked[child_first]:
                check(child_first)
            if not checked[sib_next]:
                check(sib_next)
            if not checked[sib_prev]:
                check(sib_prev)
        if checked[i]:
            i = (i + 1) % len(checked)
        else:
            check(i)
    return C, top

def label_mask(mask):
    structure = [[1,1,1],[1,1,1],[1,1,1]]
    labeled_mask, num_labels = ndimage.label(mask, structure=structure)
    colored = color.label2rgb(labeled_mask, bg_label=0)
    return labeled_mask, num_labels, colored

def make_labeled_contour_aggregation(gray, region_masks):
    z = np.zeros_like(gray)
    z = z.astype(np.int16)
    z[:,:] = -1
    for i, mask in enumerate(region_masks):
        pass

def all_imagestrs():
    for f in os.listdir("."):
        if f.endswith(".tif"):
            yield f

def combine_thresh_masks(masks):
    scale = 255 // len(masks)
    gradient = np.zeros_like(masks[0])
    binary = np.zeros_like(masks[0])
    for m in masks:
        gradient += m // 255 * scale
        binary += m
        binary[binary == 0] = 254
        binary[binary == 255] = 0
        binary[binary == 254] = 255
    return gradient, binary


# ============================================================================
# Histogram Check Module - Image Preprocessing
# ============================================================================

def log_normalize(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(img_clahe, (5,5), 0)
    kernel = np.ones((5,5),np.uint8)
    img_opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    return img_opening

def brighten_normalize(gray):
    gray = np.float32(gray)
    gray_padded = np.pad(gray, 1, mode="extend")
    print(gray_padded.shape, gray.shape)

def rm_lf(image, rad = 39):
    """Remove low frequency components using FFT"""
    dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = image.shape
    crow, ccol = rows//2 , cols//2
    
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-rad:crow+rad, ccol-rad:ccol+rad] = 1
    
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    lpf = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    
    h,l = np.max(lpf), np.min(lpf)
    lpf = (lpf - l) / (h - l)
    return lpf

def apply_multi_low_pass(gray):
    im = rm_lf(gray)
    k = 20
    t = 100
    r = 70
    for rad in range(k, k + t):
        im += rm_lf(gray, rad)
    im = im + np.float32(gray) / 255 * r
    im = im / (t + r)
    return np.uint8(im * 255)


# ============================================================================
# Pyramid Class
# ============================================================================

class Pyramid:
    def __init__(self, IA, pos :np.ndarray, id = None):
        self.positions = [pos]
        self.IA = IA
        self.cross_lengths = [5,5,5,5]
        self.id = id
        self.cross_mask = None
        self.mean_position = pos

    def set_cross_lengths(self, lengths):
        self.cross_lengths = lengths

    def get_center(self):
        return self.mean_position

    def absorb(self, other):
        if other is None:
            return
        if other is self:
            return
        self.positions += other.positions
        self.mean_position = np.mean(self.positions, axis=0)

    def get_mask(self, canvas = None, color=None, meanonly = False, lengths = None):
        if meanonly:
            return self.get_cross_mask(self.mean_position[0], self.mean_position[1], color=color, canvas=canvas, lengths = lengths)
        for pos in self.positions + [self.mean_position]:
            canvas = self.get_cross_mask(pos[0], pos[1], canvas = canvas, color=color, lengths = lengths)
        return canvas

    def get_line_endpoint(self, length, i, rotation = None, x = None, y = None):
        if length < 0:
            raise ValueError("Length must be positive")
        _x, _y = self.get_center()
        x = _x if x is None else x
        y = _y if y is None else y
        rotation = self.IA.rotation if rotation is None else rotation

        endpoint = np.float32((x + length * np.cos(rotation + i * np.pi / 2), y + length * np.sin(rotation + i * np.pi / 2)))
        return endpoint

    def get_cross_mask(self, x = None, y = None, color=None, lengths = None, rotation = None, thickness = 1, canvas = None):
        _x, _y = self.get_center()
        x = _x if x is None else x
        y = _y if y is None else y
        x_int = int(x)
        y_int = int(y)
        rotation = self.IA.rotation if rotation is None else rotation
        if canvas is None:
            canvas = np.zeros_like(self.IA.gray, dtype=np.int16) - 1
        if color is None:
            color = self.id if color is None else self.color
        lengths = self.cross_lengths if lengths is None else lengths
        for i in range(4):
            endpoint = self.get_line_endpoint(lengths[i], i, rotation)
            line(canvas, (x_int, y_int), endpoint,
                    color=color, thickness=thickness)
        return canvas

    def is_inside_box(self, box):
        x, y = self.get_center()
        box = [np.min(box[:,0]), np.min(box[:,1]), np.max(box[:,0]), np.max(box[:,1])]
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]


# ============================================================================
# Segmenting Helpers
# ============================================================================

rng = np.random.default_rng()

def get_weighted_mean(unique_points, unique_point_count, magnitudes, gray):
    brightnesses = np.float32(gray[unique_points[:, 0], unique_points[:, 1]].reshape(-1))
    min_brightness, max_brightness = bn.nanmin(brightnesses), bn.nanmax(brightnesses)
    if max_brightness == min_brightness:
        if magnitudes.shape[0] == 0:
            return 0
        else:
            return np.mean(magnitudes)
    brightness_rank = (brightnesses - min_brightness + 1) / (max_brightness - min_brightness)
    weights = brightness_rank * unique_point_count
    mean = np.average(magnitudes, axis=0, weights=weights)
    return mean

def get_weighted_median(unique_points, unique_point_count, magnitudes, gray):
    brightnesses = np.float32(gray[unique_points[:, 0], unique_points[:, 1]].reshape(-1))
    min_brightness, max_brightness = bn.nanmin(brightnesses), bn.nanmax(brightnesses)
    brightness_rank = (brightnesses - min_brightness + 1) / (max_brightness - min_brightness)
    weights = brightness_rank * unique_point_count
    magnitude_args = np.argsort(magnitudes)
    target = sum(weights)
    cumsum = np.cumsum(weights[magnitude_args])
    median_index = np.searchsorted(cumsum, target / 2)
    median = magnitudes[magnitude_args[median_index]]
    return median

def calculate_withlen_and_orthlen(startpoint, direction_vector, points, pointints, canvas, gray, active, new_points, threshold_percentile = 50):
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    pointints_active = pointints[active]
    
    # Clamp points to be within image bounds
    pointints_active[:, 0] = np.clip(pointints_active[:, 0], 0, gray.shape[0] - 1)
    pointints_active[:, 1] = np.clip(pointints_active[:, 1], 0, gray.shape[1] - 1)
    
    unique_points, unique_point_indicies, unique_point_count = np.unique(pointints_active, axis=0, return_index=True, return_counts=True)
    brightness_levels = gray[pointints_active[:, 0], pointints_active[:, 1]]
    try:
        brightness_threshold = np.percentile(brightness_levels, threshold_percentile, interpolation='lower')
    except:
        print(np.sum(brightness_levels))
        input("brightness_threshold error")
        brightness_threshold = 0
    
    bright_enough = (brightness_levels >= brightness_threshold).reshape(-1)
    bright_enough_unique = (brightness_levels[unique_point_indicies] >= brightness_threshold).reshape(-1)
    
    kept_pointints = pointints_active[bright_enough, :]
    canvas[kept_pointints[:, 0], kept_pointints[:, 1], :] = np.uint8((0, 0, 255))
    kept_half = bright_enough
    
    orth_dists = scalars_to_line_orth(startpoint, direction_vector, pointints_active).reshape(-1)
    orth_dists_unique = orth_dists[unique_point_indicies[bright_enough_unique]]
    magnitudes = scalars_to_with_line(startpoint, direction_vector, pointints_active).reshape(-1)
    magnitudes_unique = magnitudes[unique_point_indicies[bright_enough_unique]]
    
    if len(magnitudes) == 0:
        return 0, 0

    medweight = 4
    meanweight = 9
    withlen_median = get_weighted_median(unique_points[bright_enough_unique], unique_point_count[bright_enough_unique], magnitudes_unique, gray)
    withlen_mean = get_weighted_mean(unique_points[bright_enough_unique], unique_point_count[bright_enough_unique], magnitudes_unique, gray)
    orthlen_median = get_weighted_median(unique_points[bright_enough_unique], unique_point_count[bright_enough_unique], orth_dists_unique, gray)
    orthlen_mean = get_weighted_mean(unique_points[bright_enough_unique], unique_point_count[bright_enough_unique], orth_dists_unique, gray)
    
    DISTDIFF_MIN = 8
    split_leg, split_orth = None, None
    orth = np.array([direction_vector[1], -direction_vector[0]])
    withlen = (withlen_mean * meanweight + withlen_median * medweight) / (medweight + meanweight)
    orthlen = (orthlen_mean * meanweight + orthlen_median * medweight) / (medweight + meanweight)
    
    medpoint = np.array((withlen_median, orthlen_median))
    meanpoint = np.array((withlen_mean, orthlen_mean))
    thispoint = np.array((withlen, orthlen))
    
    if np.linalg.norm(thispoint) * 2 < np.linalg.norm(medpoint - meanpoint) > 1:
        if rng.random() < 0.5:
            withlen = withlen_mean * (1 + rng.random())
        else:
            withlen = withlen_median * (1 + rng.random())
        if rng.random() < 0.5:
            orthlen = orthlen_mean * (1 + rng.random())
        else:
            orthlen = orthlen_median * (1 + rng.random())
    
    if withlen == np.inf or withlen == -np.inf or withlen == np.nan or withlen == float("nan") or len(magnitudes) == 0:
        print("withlen error", withlen, withlen_mean, withlen_median, len(magnitudes))
        return 0, 0
    
    arrow(canvas, startpoint, startpoint + withlen_mean * direction_vector, (200, 80, 0), 3)
    arrow(canvas, startpoint, startpoint + withlen_median * direction_vector, (25, 20, 150), 2)
    arrow(canvas, startpoint + withlen_mean * direction_vector, startpoint + orthlen_mean * orth + withlen_mean * direction_vector, (200, 80, 0), 3)
    arrow(canvas, startpoint + withlen_median * direction_vector, startpoint + orthlen_median * orth + withlen_median * direction_vector, (25, 20, 150), 2)
    
    return withlen, orthlen


# ============================================================================
# ImageAnalysis Class
# ============================================================================

class ImageAnalysis():
    def __init__(self, imgstr):
        self.imgstr = imgstr
        if not os.path.exists(imgstr):
            self.imgstr = 'images_in/' + imgstr
        
        self.fullimg :np.ndarray = cv2.imread(self.imgstr)
        
        if self.fullimg is None:
            raise FileNotFoundError(f"Could not load image from '{imgstr}' or 'images_in/{imgstr}'. "
                                    f"Please check the file path and ensure the image exists.")
        
        self.img :np.ndarray = self.fullimg[:-77,:]
        self.titlebar :np.ndarray = self.fullimg[-76:,:]
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
        self.img = np.dstack((self.gray, self.gray, self.gray))
        self.imgwidth = self.img.shape[1]
        self.guibar_original = np.zeros((200, self.imgwidth, 3), dtype=np.uint8) + 255
        self.guibar = self.guibar_original.copy()
    
        self.render_canvas = self.zeros_color()
        self.thresh_value = 110
        self.contour_masks = None
        self.region_contexts = []
        self.rotation = 0
        self.cross_value_landscape_stored = None
        self.claimed_territory_stored = None
        self.reset_canvas()
        self.laplacian_stored = None
        
        # Create window (without topmost)
        self.window = cv2.namedWindow('Image Analysis', cv2.WINDOW_NORMAL)
        
        self.capture = None
        self.pyramids = []
        self.pyramids_by_id = {}
        self.last_show = self.img.copy()
        self.prominence = 35

    def save_clipped(self):
        name = self.imgstr.split('/')[-1]
        name, ext = name.split('.')
        clipped_path =f'clipped/{name}_clipped.{ext}'
        clipped_path_gray =f'clipped/{name}_clipped_gray.{ext}'
        cv2.imwrite(clipped_path, self.img)
        cv2.imwrite(clipped_path_gray, self.gray)
        return clipped_path, clipped_path_gray

    def show(self, img = None, ms=1, frames = 1, dont_save_last = False, titlebar = None):
        if img is None:
            img = self.last_show 
        elif not dont_save_last:
                self.last_show = img
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = np.dstack((img, img, img)).reshape(-1, self.imgwidth, 3)
        titlebar = self.titlebar if titlebar is None else titlebar
        img = np.concatenate((img, titlebar, self.guibar), axis = 0)
        cv2.imshow('Image Analysis', img)
        if self.capture is not None:
            self.capture += [img.copy()] * (frames)
        print("show", ms, "ms")
        if ms > 0:
            ms = 1
            cv2.waitKey(ms)
    
    def zeros(self):
        return np.uint8(np.zeros_like(self.gray))

    def zeros_color(self):
        return np.zeros_like(self.img)
    
    def reset_canvas(self):
        self.render_canvas = self.img.copy()

    def draw_cross(self, x, y, color=(0, 0, 255), lengths = [5,5,5,5], rotation = 0):
        x_int = int(x)
        y_int = int(y)
        for i in range(4):
             cv2.line(self.render_canvas, (x_int, y_int), 
                    (x_int + int(lengths[i] * np.cos(rotation + i * np.pi / 2)), 
                     y_int + int(lengths[i] * np.sin(rotation + i * np.pi / 2))),
                    color, 1)

    def render_display(self, **kwargs):
        self.show(self.render_canvas, **kwargs)

    def consolidate_pyramids(self):
        """Absorb pyramids that overlap"""
        pyramid_map = np.zeros_like(self.gray, dtype=np.int16) - 1
        for pyr in self.pyramids:
            overlap = np.unique(pyramid_map[pyr.get_mask() > -1])
            overlap = overlap[overlap > -1]
            for overlapping in overlap:
                pyr.absorb(self.pyramids_by_id[overlapping])
                self.pyramids_by_id[overlapping] = None
            pyramid_map[pyr.get_mask() > -1] = pyr.id
        self.pyramids = [p for p in self.pyramids_by_id.values() if p is not None]
        self.pyramids_by_id = {p.id : p for p in self.pyramids}

    def reindex_pyramids(self):
        for i, pyr in enumerate(self.pyramids):
            pyr.id = i
            pyr.mask[pyr.mask > 0] = i
        self.pyramids_by_id = {p.id : p for p in self.pyramids}

    def segment(self, segment_density = 18):
        """Detect pyramids """
        points = segment(self, segment_density)
        pyramids = [Pyramid(self, p, i) for i,  p in enumerate(points)]
        pyramid_map = np.zeros_like(self.gray, dtype=np.int16) - 1
        
        for pyr in pyramids:
            overlap = np.unique(pyramid_map[pyr.get_mask() > -1])
            overlap = overlap[overlap > -1]
            for overlapping in overlap:
                pyr.absorb(pyramids[overlapping])
                pyramids[overlapping] = None
            pyramid_map[pyr.get_mask() > -1] = pyr.id
        self.pyramids = [p for p in pyramids if p is not None]
        self.pyramids_by_id = {p.id : p for p in self.pyramids}

    def draw_pyramids(self, meanonly=True):
        self.reset_canvas()
        self.render_canvas = self.img.copy()
        for pyr in self.pyramids:
            pyr.get_mask(canvas=self.render_canvas, color = (0, 0, 255), meanonly=meanonly)
        self.show(self.render_canvas, ms=50, frames = 20)

    def get_leg_lengths(self):
        """Calculate leg lengths"""
        direction_vectors = [(np.cos(self.rotation + i * np.pi / 2), np.sin(self.rotation + i * np.pi / 2)) for i in range(4)]
        canvas = self.img.copy()
        ll = canvas.shape[0] + canvas.shape[1]
        linspace = np.linspace(0, ll, ll)
        for pyr in self.pyramids:
            for i, direction_vector in enumerate(direction_vectors):
                start_point = pyr.mean_position
                length_max, p = 0, start_point
                direction_vector = np.array(direction_vector)
                direction_vector = direction_vector / np.linalg.norm(direction_vector)
                while 0 < p[0] < self.img.shape[0] - 1 and 0 < p[1] < self.img.shape[1]- 1 :
                    p = start_point + length_max * direction_vector
                    length_max += 1
                
                length_max -= 2
                length_max = max(length_max, 0)
                check_len = -1
                
                while check_len < length_max:
                    check_len = min(check_len + 100, length_max)
                    samples = linspace[:check_len]
                    sample_locations = start_point.reshape(1, -1) + direction_vector.reshape(1, -1) * samples.reshape(-1, 1)
                    samples = interpolate.interpn((np.arange(self.img.shape[0]), np.arange(self.img.shape[1])), self.gray, sample_locations)
                    
                    samples = 255 - samples

                    peaks = signal.find_peaks(samples, prominence = self.prominence)
                    print(peaks)
                    
                    peaks = peaks[0]
                    peak = peaks[0] if len(peaks) > 0 else check_len
                    pyr.cross_lengths[i] = peak
                    if peak < check_len - 1:
                        break
            self.draw_pyramids()
 
    def to_json(self):
        self.reindex_pyramids()
        data = {}
        data['pyramids'] = [pyr.mean_position.to_json() for pyr in self.pyramids]
        data['rotation'] = self.rotation
        data['imgstr'] = self.imgstr


def IA_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    IA = ImageAnalysis(data['imgstr'])
    IA.rotation = data['rotation']
    IA.pyramids = [Pyramid.from_json(p, IA) for p in data['pyramids']]
    IA.pyramids_by_id = {p.id : p for p in IA.pyramids}
    return IA


# ============================================================================
# ImageSegmenter Class (from grad_like.py and segmenting.py)
# ============================================================================

class ImageGradStats:
    """Image gradient statistics for pyramid detection"""
    def __init__(self, IA: ImageAnalysis, stepsize = 0.007161179):
        self.IA: ImageAnalysis = IA
        self.laplacian_stored = None
        self.sobel_stored = None
        self.filter2d_stored = None
        self.filter2d_kernel_stored = None
        self.gray = self.IA.gray.copy()
        self.default_iterations = 100
        self.default_stepsize = stepsize
        self.default_stepf = self.stepf_generate(self.default_stepsize)
        self.default_interval = 10
        self.blur_gray()
        
    def reset_gray(self):
        self.gray = self.IA.gray.copy()

    def blur_gray(self, ksize = 5):
        self.gray: np.ndarray = apply_multi_low_pass(self.gray)
        self.gray: np.ndarray = cv2.GaussianBlur(self.gray, (ksize, ksize), 0)

    def sobel(self):
        return cv2.Sobel(self.gray, cv2.CV_64F, 1, 1, ksize=5)
    
    def dgrad(self, gray = None):
        gray = np.float64(self.gray) if gray is None else gray
        x = (np.roll(gray, 1, axis=0) - np.roll(gray, -1, axis=0))/2
        y = (np.roll(gray, 1, axis=1) - np.roll(gray, -1, axis=1))/2
        return np.dstack((x, y))
    
    def npdgrad(self):
        gray = np.float64(self.gray)
        x = np.gradient(gray, axis=0)
        y = np.gradient(gray, axis=1)
        return np.dstack((x, y))
    
    def sobelxy(self, ksize = 5):
        b = np.float64(self.gray)
        x = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=ksize)
        y = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=ksize)
        x_b = cv2.GaussianBlur(x[:,:], (5,5), 0)
        y_b = cv2.GaussianBlur(y[:,:], (5,5), 0)
        return np.dstack((y_b, x_b))

    def get_laplacian(self):
        blur = cv2.GaussianBlur(self.gray, (3,3), 0)
        self.laplacian_stored = cv2.Laplacian(blur, cv2.CV_64F)
        return self.laplacian_stored

    def active_points(self, points, active=None):
        """Check which points are still within bounds"""
        if active is None:
            active = np.ones(points.shape[0], dtype=bool)
        active[points[:,0] < 0] = False
        active[points[:,0] >= self.gray.shape[0]] = False
        active[points[:,1] < 0] = False
        active[points[:,1] >= self.gray.shape[1]] = False
        return active

    @staticmethod
    def interpolate_color(colors):
        if not isinstance(colors, np.ndarray):
            colors = np.uint8(colors)
        def interpolate_color_func_over_iterations(iterations):
            def interpolate_color_func(i):
                t = i / iterations
                return colors[0] * (1 - t) + colors[1] * t
            return interpolate_color_func
        return interpolate_color_func_over_iterations

    @staticmethod
    def stepf_generate(stepsize):
        if type(stepsize) in (float, int):
            stepsize = np.array((stepsize, stepsize))
        stepf = ImageGradStats.interpolate_color(stepsize)
        return stepf

    def hillroll_points_iterate_velocity(self, landscape, points, iterations=100, stepsize=0.001, 
                                        canvas=None, color=None, normalize=False, 
                                        velocities=None, velocity_decay=0.9, addcolors=False, show=False):
        """Iterate points rolling on gradient landscape with velocity"""
        assert points.shape[1:] == (2,)
        points = np.float64(points)
        canvas = np.zeros_like(landscape) if canvas is None else canvas
        pointints = points.astype(np.int16)
        
        if velocities is None:
            velocities = np.zeros_like(points)
        else:
            velocities = np.ones_like(points) * velocities
            
        if callable(stepsize):
            # stepsize is already a function factory, call it with iterations
            stepsize_func = stepsize(iterations)
        else:
            # Generate function from numeric stepsize
            stepf = self.stepf_generate(stepsize)
            stepsize_func = stepf(iterations)
        
        if callable(color):
            # color is already a function factory, call it with iterations
            color_func = color(iterations)
        else:
            # Generate function from color tuple
            colors = self.interpolate_color(((0, 0, 0), (0, 255, 0)))
            color_func = colors(iterations)
        
        active = np.ones_like(points[:,0]).astype(bool)
        active = self.active_points(points, active)
        
        for i in range(iterations):
            pointints[active,:]
            dirs = landscape[pointints[active,0],pointints[active,1], :] * stepsize_func(i)
            velocities[active,:] *= velocity_decay
            velocities[active,:] += dirs
            points[active,:] += velocities[active,:]
            
            active = self.active_points(points, active)
            pointints[active, :] = np.dstack((np.clip(points[active,0], 0, landscape.shape[0] - 1), 
                                              np.clip(points[active,1], 0, landscape.shape[1] - 1)))
            if addcolors:
                canvas[pointints[active,0], pointints[active,1], 1:] += color_func(i)[1:].astype(canvas.dtype)
            else:
                canvas[pointints[active,0], pointints[active,1], 1:] = np.uint8(color_func(i)[1:])
            if show:
                self.IA.show(canvas)
        
        return {'points': points, 'velocities': velocities, 'canvas': canvas, 
                'active': self.active_points(pointints), "pointints": pointints}
    
    def hillroll_points(self, points=None, up=False, both=False, **kwargs):
        """Main hillroll function"""
        colorsUp = self.interpolate_color(((0, 0, 0), (0, 255, 0)))
        colorsDown = self.interpolate_color(((0, 0, 0), (0, 0, 255)))
        
        interval = 10 if "interval" not in kwargs else kwargs["interval"]
        points = np.mgrid[0: self.gray.shape[0]: interval, 0: self.gray.shape[1]: interval].reshape(2, -1).T if points is None else points
        iterations = 100 if "iterations" not in kwargs else kwargs["iterations"]
        stepsize_val = self.default_stepsize if "stepsize" not in kwargs else kwargs["stepsize"]
        stepf = self.stepf_generate(stepsize_val) if "stepf" not in kwargs else kwargs["stepf"]
        nstepf = self.stepf_generate(-1 * stepsize_val) if "nstepf" not in kwargs else kwargs["nstepf"]
        dgrad = self.npdgrad()
        sobel = self.sobelxy(ksize=3) if "landscape" not in kwargs else kwargs["landscape"]
        landscape = dgrad
        landscape = (sobel + dgrad) / 2
        canvas = self.IA.img.copy() if "canvas" not in kwargs else kwargs["canvas"]
        
        # Remove stepsize from kwargs if present to avoid conflicts
        kwargs_copy = kwargs.copy()
        if 'stepsize' in kwargs_copy:
            del kwargs_copy['stepsize']
        
        if up:
            d = {'landscape': landscape, 'points': points, 'iterations': iterations, 
                 'canvas': canvas, 'color': colorsUp, 'stepsize': stepf}
            d.update(kwargs_copy)
            return self.hillroll_points_iterate_velocity(**d)
        if both or not up:
            d = {'landscape': landscape, 'points': points, 'iterations': iterations, 
                 'canvas': canvas, 'color': colorsDown, 'stepsize': nstepf}
            d.update(kwargs_copy)
            return self.hillroll_points_iterate_velocity(**d)

class ImageSegmenter:
    def __init__(self, IA: ImageAnalysis):
        self.gs: ImageGradStats = ImageGradStats(IA, 0.08)
        self.IA = IA

    def est_leg_len_parallel(self, startpoints :np.ndarray, direction_vector:np.ndarray, box_rad = 4, density = 2, steps = 150, iterations = 150, box_spray_velocity = -0.75, velocity_decay = 0.9, doshow=False, threshold_percentile = 50):
        """
        Estimate leg length in a direction for pyramids with centers at startpoints[]. 
        """
        density = 2j * box_rad / density
        P = startpoints.shape[0]
        testpoints = np.mgrid[100 - box_rad : 100 + box_rad: density, 100 - box_rad : 100 + box_rad: density].reshape(2, -1).T
        stride = testpoints.shape[0]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        leglens = np.zeros_like(startpoints[:, 0])
        
        def make_points_and_velocities(startpoints):
            shape = (P * stride, 2)
            points = np.zeros(shape, dtype = np.float64)
            velocities = np.zeros(shape, dtype = np.float64)
            for i in range(P):
                points[i * stride : (i + 1) * stride, :] = np.mgrid[startpoints[i, 0] - box_rad : startpoints[i, 0] + box_rad: density, startpoints[i, 1] - box_rad : startpoints[i, 1] + box_rad: density].reshape(2, -1).T
            velocities_sample = dirs_to_point(points[0:stride, :], startpoints[0, :]) * box_spray_velocity
            velocities.reshape(P, stride, 2)[:, :, :] = velocities_sample
            return points, velocities

        def velocity_iteration(i, d):
            velocities_to_core = 0 * dirs_to_point(startpoints.reshape(P, 1, 2), d["points"].reshape(P, stride, 2)).reshape(P * stride, 2)
            velocities_toward_line = 0 * dirs_to_line_orth(startpoints.reshape(P, 1, 2), direction_vector, d["points"].reshape(P, stride, 2)).reshape(P * stride, 2)
            return 0
        
        new_points = []
        points, velocities = make_points_and_velocities(startpoints)
        landscape = self.gs.sobelxy(ksize=3) / 5
        landscape_cross = np.dstack((landscape[:,:,1], -1 * landscape[:,:,0]))
    
        d = {'points': points, 'velocities': velocities, 'landscape' : landscape, "show":doshow}
    
        steps = 1
        for i in range(steps):
            d = self.gs.hillroll_points(iterations = iterations//steps, velocity_decay = velocity_decay, up = True, **d)
            self.IA.show(d["canvas"], frames = 1)
            d["velocities"] += velocity_iteration(i, d)
        
        leglens_out = np.zeros_like(leglens, dtype = np.float64)
        orthlens_out = np.zeros_like(leglens, dtype = np.float64)
        
        points = d["points"].reshape(P, stride, 2)
        pointints = d["pointints"].reshape(P, stride, 2)
        active = d["active"].reshape(P, stride)
        
        for i in range(P):
            l, o = calculate_withlen_and_orthlen(startpoint=startpoints[i], direction_vector=direction_vector,     
                                                      points=points[i], pointints=pointints[i], canvas=d["canvas"], 
                                                      gray=self.IA.gray, active=active[i], new_points=new_points,
                                                      threshold_percentile = threshold_percentile)
            leglens_out[i] = l 
            orthlens_out[i] = o
            if doshow:
                self.IA.show(d["canvas"], ms = 1, frames = 2)

        self.IA.show(d["canvas"], ms = 10, frames = 2)
        return leglens_out, orthlens_out, new_points


def segment(IA: ImageAnalysis, density = 40): 
    gray = IA.gray
    segmenter = ImageSegmenter(IA)
    frames =[]
    IA.capture = frames
    segmenter.gs.gray = gray
    velocity_scales = [1] * 50
    
    startpoints = np.array(np.meshgrid(np.arange(0, IA.gray.shape[0], density), np.arange(0, IA.gray.shape[1], density))).T.reshape(-1, 2)
    direction_vector = np.array([1, 1])
    
    MAX_START_POINTS = 10000000
    MIN_BRIGHTNESS_FOR_DUPLICATE_ACCEPT = 175
    MIN_BRIGHTNESS_FOR_TRIPLICATE_ACCEPT = 135
    stability_map = np.zeros_like(IA.gray, dtype = bool)
    skipped_points = []
    stable_points = []
    
    def add_stable_points(points, reason = "unknown"):
        points = points.reshape(-1, 2)
        for i in range(points.shape[0]):
            p = points[i]
            if not stability_map[p[0], p[1]]:
                stable_points.append(p)
            stability_map[p[0], p[1]] = True
            
    k = -1
    iterations = [70, 30, 60, 40, 30, 30, 300, 250, 30, 76,76] * 10
    iterations = ([65, 56] * 1 + [500, 300, 400, 1000, 100, 150, 600]) * 2
    percentile = 80
    
    for iters in iterations:
        if len(startpoints) == 0:
            break
        leglens, orthlens, new_points = segmenter.est_leg_len_parallel(startpoints, direction_vector, iterations = iters,
                                                                 velocity_decay=0.96, box_spray_velocity=k, doshow = False,
                                                                 threshold_percentile = percentile)
        k= k * 0.98
        percentile = 1 - (1 - percentile) * 0.99
        orth = np.array([direction_vector[1], -1 * direction_vector[0]])
        orth = orth / np.linalg.norm(orth)
        startpoints = startpoints + (orthlens.reshape(-1, 1) * orth.reshape(1, 2))
        startpoints = startpoints + (leglens.reshape(-1, 1) * direction_vector.reshape(1, 2))
        both_lens = np.dstack((leglens, orthlens)).reshape(-1, 2)
        movement = np.linalg.norm(both_lens, axis = 1)
        stationary = movement < 0.1
        
        add_stable_points(np.int32(startpoints[stationary, :]), reason = "stationary")
        both_lens = both_lens[~stationary]
        startpoints = startpoints[~stationary, :]
        
        if len(new_points) > 0 and len(startpoints) < MAX_START_POINTS:
            startpoints = np.concatenate((startpoints, np.array(new_points).reshape(-1, 2)), axis = 0)
            both_lens = np.concatenate((both_lens, np.zeros_like(new_points).reshape(-1,2)), axis = 0)
        else:
            skipped_points += new_points
        
        startpoints = np.int32(startpoints)
        inbound_x = np.logical_and(startpoints[:, 0] > 0, startpoints[:, 0] < IA.gray.shape[0])
        inbound_y = np.logical_and(startpoints[:, 1] > 0, startpoints[:, 1] < IA.gray.shape[1])
        inbounds = np.logical_and(inbound_x, inbound_y)
        both_lens = both_lens[inbounds, :]
        startpoints = startpoints[inbounds,:]
        
        both_lens = both_lens[~stability_map[startpoints[:, 0], startpoints[:, 1]], :]
        startpoints = startpoints[~stability_map[startpoints[:, 0], startpoints[:, 1]]]
        
        if np.unique(startpoints, axis = 0).shape[0] < startpoints.shape[0]:
            unique, uindicies, counts = np.unique(startpoints, axis = 0, return_index=True, return_counts = True)
            duplicates = counts == 2
            brightnesses = gray[unique[:, 0], unique[:, 1]]
            removable_duplicates = np.logical_and(duplicates, brightnesses > MIN_BRIGHTNESS_FOR_DUPLICATE_ACCEPT)
            triplicates_up = counts >= 3
            removable_triplicates = np.logical_and(triplicates_up, brightnesses > MIN_BRIGHTNESS_FOR_TRIPLICATE_ACCEPT)
            removable = np.logical_or(removable_duplicates, removable_triplicates)
            
            add_stable_points(unique[removable], reason = "duplicate")
            both_lens = both_lens[uindicies, :]
            startpoints = startpoints[uindicies, :]
            both_lens = both_lens[~removable, :]
            startpoints = startpoints[~removable, :]
    
    print(stable_points)
    IA.reset_canvas()
    for p in startpoints:
        IA.draw_cross(p[1], p[0], lengths = [5,5,5,5], color = (250, 0, 0))
    for i in range(len(stable_points)):
        IA.draw_cross(stable_points[i][1], stable_points[i][0], lengths = [5,5,5,5], color = (0, 250, 0))
            
    add_stable_points(startpoints, reason = "final")
    IA.render_display(ms = 1000)
    return stable_points


# ============================================================================
# Height Calculation Functions
# ============================================================================

def pixels_to_um(gui_control, pixels: np.ndarray):
    return pixels * gui_control.um_per_pixel()

def H(L, sin_degrees: str = "45"):
    sinfactor = np.sin(np.radians(float(sin_degrees)))
    H1pre = max(L)
    H2pre = min(L)
    H3pre = (L[0] + L[1] + L[2] + L[3]) / 4
    Base_area = (L[0] + L[2])*(L[1] + L[3])/4
    H1_area = Base_area
    H2_area = Base_area
    H3_area = Base_area
    H1 = H1pre * sinfactor
    H1_weighted = H1*Base_area
    H2 = H2pre * sinfactor
    H2_weighted = H2*Base_area
    H3 = H3pre * sinfactor
    H3_weighted = H3*Base_area
    work1 = f"Maximum height = max(L1, L2, L3, L4) * sin({sin_degrees}) = {H1pre} * sin({sin_degrees}) = {H1} um"
    work2 = f"Minimum height = min(L1, L2, L3, L4) * sin({sin_degrees}) = {H2pre} * sin({sin_degrees}) = {H2} um"
    work3 = f"Average height = (L1 + L2 + L3 + L4) / 4 * sin({sin_degrees}) = {H3pre} * sin({sin_degrees}) = {H3} um"
    return ((H1, work1, H1_weighted, H1_area), (H2, work2, H2_weighted, H2_area), (H3, work3, H3_weighted, H3_area))

def height_calc(gui_control, pyramids: List[Pyramid]) -> Tuple[float, str]:
    """Calculate the height of the pyramids in the image."""
    heights = [[],[],[]]
    heights_weighted = [[],[],[]]
    work = [[], [], []]
    Base_areas = [[], [], []]
    for p in pyramids:
        L = [gui_control.pixels_to_um(l) for l in p.cross_lengths]
        H1, H2, H3 = H(L)
        heights[0].append(H1[0])
        work[0].append(H1[1])
        heights_weighted[0].append(H1[2])
        Base_areas[0].append(H1[3])
        
        heights[1].append(H2[0])
        work[1].append(H2[1])
        heights_weighted[1].append(H2[2])
        Base_areas[1].append(H2[3])
        
        heights[2].append(H3[0])
        work[2].append(H3[1])
        heights_weighted[2].append(H3[2])
        Base_areas[2].append(H2[3])
        
    return heights, work, heights_weighted, Base_areas

def make_hist(gui_control, heights: List[List[float]], work, heights_weighted, Base_areas, save: bool = True, show: bool = True):
    fig, axs = plt.subplots(len(heights), sharex=True)
    fig.suptitle("Pyramid Heights Histogram")

    titles = ["Maximum Height", "Minimum Height", "Average Height"]
    for i, ax in enumerate(axs):
        hist, bins = np.histogram(heights[i], bins=np.arange(0, 3.05, 0.05))
        ax.bar(bins[:-1], hist, width=0.05, align='edge')
        ax.set_ylabel('Counts [pyramids]')
        ax.set_title(titles[i])
        ax.text(0.05, 0.8, 'Mean = {:.2f} um'.format(np.mean(heights[i])), transform=ax.transAxes)
        ax.text(0.05, 0.7, 'Total Pyramids = {}'.format(len(heights[i])), transform=ax.transAxes)
        ax.text(0.05, 0.6, 'A.W.Mean = {:.2f} um'.format(np.sum(heights_weighted[i])/np.sum(Base_areas[i])), transform=ax.transAxes)
    plt.xlabel("Pyramid height [um]")

    if save:
        imgname = gui_control.IA.imgstr.split("/")[-1].split(".")[0]
        os.makedirs("./output/" + imgname, exist_ok=True)
        plt.tight_layout()
        plt.savefig("./output/" + imgname + "/" + "_hist.png")
        dataframe = pd.DataFrame({"Max Height (um)":heights[0], "Max Height Calculations":work[0], "Min Height (um)":heights[1], "Min Height Calculations":work[1], "Avg Height (um)":heights[2], "Avg Height Calculations":work[2]})
        dataframe.to_excel(f'./output/{imgname}/heights.xlsx', index = False)
        for i in range(len(titles)):
            height_measure = titles[i].replace(" ", "")
            w = open("./output/" + imgname + "/" + height_measure + "_hist_calculations.txt", "w")
            for j in range(len(work[i])):
                w.write("Pyramid " + str(j) + ": " + work[i][j] + "\n")
            h = open("./output/" + imgname + "/" + height_measure + "_hist_heights.txt", "w")
            for j in range(len(heights[i])):
                h.write(str(heights[i][j]) + "\n")
    if show:
        plt.show()

def make_hist_with_subplots(gui_control, heights: List[float], height_measure:str, work: List[str], heights_weighted: List[float], Base_areas: List[float], subplot_params) -> Tuple[np.ndarray, np.ndarray]:
    """Make a histogram of the heights of the pyramids."""
    heights = np.array(heights)
    heights_weighted = np.array(heights_weighted)
    Base_areas = np.array(Base_areas)
    binsize = 0.05
    topbin = max(3, np.ceil(max(heights) / binsize) * binsize)
    bins = np.arange(0, topbin, binsize)
    hist, bins = np.histogram(heights, bins=bins)
    plt.bar(bins[:-1], hist, width=binsize, align="edge")
    plt.xlabel("Pyramid height [um]")
    plt.ylabel("Counts [pyramids]")
    plt.title("Pyramid heights for " + height_measure + " measure")
    mean = np.mean(heights)
    AW_Mean = np.sum(heights_weighted)/np.sum(Base_areas)
    median = np.median(heights)
    std = np.std(heights)
    plt.text(0.9, 0.9, f"Mean: {mean:.3f}\nA.W.Mean: {AW_Mean:.3f}", transform=plt.gca().transAxes, ha='right', va='top')
    
    if True:
        imgname = gui_control.IA.imgstr.split("/")[-1].split(".")[0]
        os.makedirs("./output/" + imgname, exist_ok=True)
        plt.savefig("./output/" + imgname + "/" + height_measure + "_hist.png")
        w = open("./output/" + imgname + "/" + height_measure + "_hist_calculations.txt", "w")
        path = "./output/" + imgname + "/" + height_measure + "_hist.png"
        for i in range(len(work)):
            w.write("Pyramid " + str(i) + ": " + work[i] + "\n")
        h = open("./output/" + imgname + "/" + height_measure + "_hist_heights.txt", "w")
        for i in range(len(heights)):
            h.write(str(heights[i]) + "\n")
    if True:
        plt.show()
    return path

def save_histimg(gui_control, orig_img, pyramids: List[Pyramid]):
    heights, work, heights_weighted, Base_areas = height_calc(gui_control, pyramids)
    titles = ["Maximum Height", "Minimum Height", "Average Height"]
    paths = []
    for i in range(len(titles)):
        paths.append(make_hist_with_subplots(gui_control, heights[i], titles[i], work[i], heights_weighted[i], Base_areas[i], subplot_params=False))
    images = list(map(Image.open, paths))
    widths, heights_img = zip(*(i.size for i in images))

    total_width = max(sum(widths), orig_img.shape[1])
    max_height = max(*heights_img) + orig_img.shape[0] 
    new_im = Image.new("RGB", (total_width, max_height))
    new_im.paste(Image.fromarray(orig_img), (0, 0))
    y_offset = orig_img.shape[0]
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
    new_im.save("./output/" + gui_control.IA.imgstr.split("/")[-1].split(".")[0] + "/overlay_hist.png")


# ============================================================================
# Main Function
# ============================================================================

# ============================================================================
# GUI Control Class
# ============================================================================

# Key mappings
K_ESC, K_ENTER, K_SPACE, K_BACKSPACE = 27, 13, 32, 8
K_UP, K_DOWN, K_LEFT, K_RIGHT, K_TAB = 82, 84, 81, 83, 9

class GuiControl():
    def __init__(self, image_path_or_IA = None):
        if isinstance(image_path_or_IA, str):
            self.IA = ImageAnalysis(image_path_or_IA)
        elif isinstance(image_path_or_IA, ImageAnalysis):
            self.IA = image_path_or_IA
        else:
            self.IA = None
            
        if self.IA is None:
            file_path = select_image_file()
            if file_path:
                self.IA = ImageAnalysis(file_path)
            else:
                print("No file selected. Exiting...")
                exit()
                
        self.bottom_exclusion = 77
        self.mode_function = None
        self.selected_pyramid = None
        self.click_point = None
        self.direction_vector = [0,1]
        self.direction_vector_arrow = [[0,0],[0,0]]
    
        self.micrometer_scale_length = 1
        self.micrometer_scale_line = np.array(((23,734), (95,734)))
        self.quit, self.savedstr, self.old_str, self.keys, self.mousedrags, self.selected_pyramids = False, "", "", [], [], []
        self.guicanvas = None
        self.prev_guicanvas = self.IA.img.copy()
        self.leg_index = 0
        self.length_in_um = ""
        self.titlebar_original = self.IA.titlebar.copy()
        self.button_bounding_boxes = []
        self.legend = {}
        self.prev_menus = []
        self.saved_guibar = self.IA.guibar.copy()
        self.leg_length_step = 2
        
        # Add a "Click here to start" message on the GUI bar initially - centered
        text = ">>> CLICK HERE TO START <<<"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        # Get text size to calculate center position
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate center position
        x = (self.IA.imgwidth - text_width) // 2
        y = (self.IA.guibar.shape[0] + text_height) // 2
        
        cv2.putText(self.IA.guibar, text, (x, y), font, font_scale, (255, 0, 0), thickness)
    
    def adjust_bottom_exclusion(self, key=None, mousedrag=None):
        """Adjust how many pixels to exclude from the bottom of the image"""
        
        if key is not None and key != 0:  # Ignore key code 0
            # Use + and - keys for fine adjustment, U/D for coarse adjustment
            if key == ord('+') or key == ord('='):  # + key (with or without shift)
                self.bottom_exclusion = min(self.bottom_exclusion + 1, 200)
                self.reload_with_exclusion()
            elif key == ord('-') or key == ord('_'):  # - key
                self.bottom_exclusion = max(self.bottom_exclusion - 1, 0)
                self.reload_with_exclusion()
            elif key == ord('u') or key == ord('U'):  # U for up (increase by 10)
                self.bottom_exclusion = min(self.bottom_exclusion + 10, 200)
                self.reload_with_exclusion()
            elif key == ord('d') or key == ord('D'):  # D for down (decrease by 10)
                self.bottom_exclusion = max(self.bottom_exclusion - 10, 0)
                self.reload_with_exclusion()
            elif key == K_ENTER or key == 13:
                print(f"Bottom exclusion set to {self.bottom_exclusion} pixels")
                return self.set_up()
        
        # Create display with cutoff line
        fullimg_display = self.IA.fullimg.copy()
        if self.bottom_exclusion > 0:
            cutoff_y = fullimg_display.shape[0] - self.bottom_exclusion
            # Draw a bright line to show where the cut happens
            cv2.line(fullimg_display, (0, cutoff_y), (fullimg_display.shape[1], cutoff_y), 
                     (0, 255, 0), 3)  # Green line
            # Add text label near the line
            cv2.putText(fullimg_display, f"Cutoff at {self.bottom_exclusion} px from bottom", 
                       (10, cutoff_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Store this as the display to show
        self.guicanvas = fullimg_display
        
        return {
            "+/=": "Increase exclusion by 1 pixel",
            "-": "Decrease exclusion by 1 pixel",
            "U": "Increase exclusion by 10 pixels",
            "D": "Decrease exclusion by 10 pixels",
            "Enter": "Confirm",
            "ESC": "Previous menu"
        }, [("Bottom exclusion (pixels)", "bottom_exclusion")]

    def reload_with_exclusion(self):
        """Reload the image with current exclusion setting"""
        # Store current image path
        imgpath = self.IA.imgstr
        
        # Reload the full image
        fullimg = cv2.imread(imgpath)
        if fullimg is None:
            print(f"Error reloading image from {imgpath}")
            return
        
        # Apply exclusion
        if self.bottom_exclusion > 0:
            self.IA.img = fullimg[:-self.bottom_exclusion, :]
            self.IA.titlebar = fullimg[-self.bottom_exclusion:, :]
        else:
            self.IA.img = fullimg
            self.IA.titlebar = np.zeros((76, fullimg.shape[1], 3), dtype=np.uint8)
        
        self.IA.fullimg = fullimg
        self.IA.gray = cv2.cvtColor(self.IA.img, cv2.COLOR_BGR2GRAY)
        self.IA.img = np.dstack((self.IA.gray, self.IA.gray, self.IA.gray))
        
        # Update related attributes
        self.IA.imgwidth = self.IA.img.shape[1]
        self.IA.render_canvas = self.IA.zeros_color()
        self.IA.reset_canvas()
        self.titlebar_original = self.IA.titlebar.copy()
        self.guicanvas = None
        self.prev_guicanvas = self.IA.img.copy()
    
    def reload_image(self, key=None, mousedrag=None):
        """Reload or select a new image"""
        if key is not None or mousedrag is not None:
            # User confirmed, select new image
            file_path = select_image_file()
            if file_path:
                print(f"Loading new image: {file_path}")
                # Create new ImageAnalysis instance
                self.IA = ImageAnalysis(file_path)
                # Reset all state
                self.selected_pyramid = None
                self.click_point = None
                self.selected_pyramids = []
                self.guicanvas = None
                self.prev_guicanvas = self.IA.img.copy()
                self.leg_index = 0
                self.titlebar_original = self.IA.titlebar.copy()
                self.saved_guibar = self.IA.guibar.copy()
                # Reset scale settings (optional - remove if you want to keep scale)
                self.micrometer_scale_length = 1
                self.micrometer_scale_line = np.array(((23,734), (95,734)))
                print("New image loaded successfully!")
                return self.default_mode()
            else:
                print("No file selected, keeping current image")
                return self.default_mode()
        
        # Show confirmation prompt - extract just the filename
        return {"Enter": "Select new image", "ESC": "Cancel"}, [("Current image", lambda: os.path.basename(self.IA.imgstr))]

    def mouse_event_callback(self, action, x, y, flags, *userdata):
        if action == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (y, x)
        elif action == cv2.EVENT_LBUTTONUP:
            if self.click_point is not None:
                dragtup = np.array((self.click_point, (y, x)), dtype = np.int32)
                self.click_point = None
                if self.mode_function:
                    self.mode_function(mousedrag = dragtup)
                    self.update_display(*self.mode_function())
            
    def control_loop(self):
        cv2.setMouseCallback("Image Analysis", self.mouse_event_callback)
        k = -1
        nextmode = None
        first_keypress = True  # Flag to track first keypress
        
        while True:
            if self.mode_function == None:
                self.mode_function = self.default_mode
            if nextmode is not None:
                # Reset add mode when changing to a different mode
                if nextmode != self.mode_function:
                    self._add_pyramid_mode = False
                ls = nextmode()
                if ls is not None:
                    legend, state = ls
                    self.mode_function = nextmode
                else:
                    legend, state = self.mode_function()
            else:
                legend, state = self.mode_function()
            
            # Keep the "CLICK HERE TO START" message until first keypress
            if first_keypress:
                # Don't update display yet, keep the initial message
                pass
            else:
                self.update_display(legend, state)
            
            k = cv2.waitKey(0)
            
            # Check if window was closed
            try:
                if cv2.getWindowProperty('Image Analysis', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break
            
            # Remove the initial message on first keypress
            if first_keypress:
                first_keypress = False
                self.IA.guibar = self.IA.guibar_original.copy()  # Clear the message
                self.update_display(legend, state)  # Now show the actual menu
            
            # Check for global ESC first, but let mode handle it in special cases
            handle_esc_globally = True
            
            if k == K_ESC:
                # Check if we're in add mode - if so, let pyramid_edit handle it
                if hasattr(self, '_add_pyramid_mode') and self._add_pyramid_mode:
                    handle_esc_globally = False
            
            if not handle_esc_globally:
                # Let the mode function handle ESC
                result = self.mode_function(key=k)
                if result is not None:
                    legend, state = result
                    self.update_display(legend, state)
            elif k == K_ESC:
                # Handle ESC globally - go back to main menu
                nextmode = None
                self.prev_menus = []
                self._add_pyramid_mode = False
                self.mode_function = self.default_mode
            elif k == ord('q') and self.mode_function == self.default_mode:
                # Quit from main menu
                break
            else:
                # First try to get a mode change
                nextmode = self.select_mode(legend, k)
                # If no mode change, pass key to current mode function
                if nextmode is None:
                    result = self.mode_function(key=k)
                    if result is not None:
                        legend, state = result
                        self.update_display(legend, state)
    
        cv2.destroyAllWindows()
    
    def update_display(self, legend, state):
        self.legend = legend
        text_list = self.set_guibar_text(legend, state)
        self.show_pyramids()
        self.saved_guibar = self.IA.guibar.copy()

    def select_mode(self, legend, key):
        # Try both uppercase and lowercase versions
        key_char_upper = chr(key).upper() if key < 256 else None
        key_char_lower = chr(key).lower() if key < 256 else None
        
        for k, item in legend.items():
            if k == key_char_upper or k == key_char_lower or (len(k) == 1 and ord(k) == key):
                # Only return if it's callable (a mode function)
                if callable(item):
                    return item
                # If it's a tuple, extract the function
                elif isinstance(item, tuple) and len(item) >= 2 and callable(item[1]):
                    return item[1]
        return None

    def set_guibar_text(self, legend, state):
        self.IA.guibar = self.IA.guibar_original.copy()
        y_offset_left = 20
        y_offset_right = 20
        text_list = []
        
        # Display legend on the LEFT side
        for key, item in legend.items():
            # Handle both tuple format (desc, function) and direct function/string
            if isinstance(item, tuple) and len(item) >= 2:
                desc_text = item[0]  # First element is description string
            elif callable(item):
                desc_text = item.__name__.replace('_', ' ').title()
            else:
                desc_text = str(item) if item else ""
            
            # Skip empty descriptions
            if not desc_text or desc_text.strip() == "":
                continue
                
            text = f"{key}: {desc_text}"
            cv2.putText(self.IA.guibar, text, (10, y_offset_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset_left += 20
            text_list.append(text)
        
        # Display state on the RIGHT side
        right_x_position = self.IA.imgwidth - 300  # 300 pixels from the right edge
        for s in state:
            if isinstance(s, str):
                label = s
                if hasattr(self, s):
                    value = getattr(self, s)
                else:
                    value = "N/A"
            elif isinstance(s, tuple) and len(s) == 2:
                label, attr_or_func = s
                # Check if second element is callable
                if callable(attr_or_func):
                    try:
                        value = attr_or_func()
                    except Exception as e:
                        value = f"Error: {e}"
                # Check if it's a string attribute name
                elif isinstance(attr_or_func, str):
                    value = getattr(self, attr_or_func) if hasattr(self, attr_or_func) else "N/A"
                else:
                    value = str(attr_or_func)
            else:
                label = str(s)
                value = "N/A"
            
            # If value itself is callable, try to call it
            if callable(value):
                try:
                    value = value()
                except:
                    value = "N/A"
            
            text = f"{label}: {value}"
            cv2.putText(self.IA.guibar, text, (right_x_position, y_offset_right), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset_right += 20
        
        return text_list

    def show_pyramids(self):
        # Check if we're in a special display mode (like bottom exclusion adjustment)
        if self.guicanvas is not None and self.guicanvas.shape[0] == self.IA.fullimg.shape[0]:
            # We're showing the full image (e.g., in bottom exclusion mode)
            # Just show it with the GUI bar, don't draw pyramids
            prevrender = self.IA.last_show.copy()
            display = np.concatenate((self.guicanvas, self.IA.guibar), axis=0)
            cv2.imshow('Image Analysis', display)
            self.prev_guicanvas = self.guicanvas
            self.guicanvas = None
            self.IA.last_show = prevrender
            return
        
        # Normal pyramid display mode
        self.IA.draw_pyramids()
        if self.guicanvas is None:
            self.guicanvas = self.IA.last_show.copy()
        for sp in self.selected_pyramids:
            sp.get_mask(self.guicanvas, color = (200, 200, 0), meanonly = True)
        if self.leg_index is not None:
            l = [0, 0, 0, 0]
            l[self.leg_index] = 1
            l = np.array(l)
            for sp in self.selected_pyramids:
                lengths = np.array(sp.cross_lengths) * l
                sp.get_mask(self.guicanvas, color = (200, 0, 200), meanonly = True, lengths = lengths)
        prevrender = self.IA.last_show.copy()
        self.IA.show(self.guicanvas)
        self.prev_guicanvas = self.guicanvas
        self.guicanvas = None
        self.IA.last_show = prevrender

    def default_mode(self, key=None, mousedrag=None):
        """Default mode - main menu"""
        legend = {
            "S": ("Set Up", self.set_up),
            "A": ("Automated", self.computations_menu),
            "M": ("Manual edit", self.pyramid_edit),
            "R": ("Reload Image", self.reload_image),
            "Q": "Quit"
        }
        state = [("Pyramids", "pyramid_count")]
        return legend, state

    def set_up(self, key=None, mousedrag=None):
        """Set up the image analysis"""
        legend = {
            "M": self.set_micrometer_scale,
            "D": self.set_direction,
            "B": self.adjust_bottom_exclusion,
            "ESC": "Previous menu"
        }
        return legend, [("Pyramids", "pyramid_count")]

    def segment_pyramids(self, key=None, mousedrag=None):
        """Segment pyramids in the image"""
        if key is not None or mousedrag is not None:
            # Already called, return to menu
            return None
        density = 20  # default density
        print(f"Segmenting with density={density}...")
        self.IA.segment(density)
        print(f"Found {len(self.IA.pyramids)} pyramids")
        return self.computations_menu()

    def pyramid_count(self):
        return len(self.IA.pyramids)

    def selected_pyramid_count(self):
        return len(self.selected_pyramids)

    def pyramid_edit(self, key=None, mousedrag=None):
        """Manual edit pyramids"""
        
        # Check if we're in "add mode"
        add_mode = hasattr(self, '_add_pyramid_mode') and self._add_pyramid_mode
        
        if key is not None:
            if key == ord("a") or key == ord("A"):
                # Only allow entering add mode if not already in it
                if not add_mode:
                    self._add_pyramid_mode = True
                    add_mode = True
            elif key == K_ESC and add_mode:
                # Exit add mode (but stay in manual edit menu)
                self._add_pyramid_mode = False
                self.selected_pyramids = []  # Clear all selections when exiting add mode
                # Return the normal manual edit menu - control_loop will see this return and not process ESC
                return self.pyramid_edit()
        
        if mousedrag is not None:
            if add_mode:
                # Add a new pyramid
                pos = mousedrag[0]
                length = max(np.linalg.norm(mousedrag[1] - mousedrag[0]), 4)
                self.IA.pyramids += [Pyramid(self.IA, pos, id=len(self.IA.pyramids))]
                self.IA.pyramids[-1].set_cross_lengths([length, length, length, length])
                self.selected_pyramids = [self.IA.pyramids[-1]]
                # Don't exit add mode - stay in it to add more pyramids
            else:
                # Select pyramids or unselect if clicking on empty area
                selected = []
                for p in self.IA.pyramids:
                    if p.is_inside_box(mousedrag):
                        selected += [p]
                # If nothing selected, unselect all (clicking on empty area)
                self.selected_pyramids = selected
        
        # Handle key presses for pyramid editing (only when not in add mode)
        if key is not None and len(self.selected_pyramids) > 0 and not add_mode:
            if key == ord("e") or key == ord("E"):
                # Increase leg length
                for p in self.selected_pyramids:
                    p.cross_lengths[self.leg_index] = max(p.cross_lengths[self.leg_index] + self.leg_length_step, 1)
            elif key == ord("q") or key == ord("Q"):
                # Decrease leg length
                for p in self.selected_pyramids:
                    p.cross_lengths[self.leg_index] = max(p.cross_lengths[self.leg_index] - self.leg_length_step, 1)
            elif key == ord("w") or key == ord("W"):
                # Change selected leg
                self.leg_index = (self.leg_index + 1) % 4 if self.leg_index is not None else 0
            elif key == ord("d") or key == ord("D"):
                # Delete pyramids
                self.IA.pyramids = [p for p in self.IA.pyramids if p not in self.selected_pyramids]
                self.IA.pyramids_by_id = {p.id: p for p in self.IA.pyramids}
                self.selected_pyramids = []
        
        # Build legend based on current state
        if add_mode:
            legend = {
                "Click": "Add pyramid at location",
                "ESC": "Exit add mode"
            }
        elif len(self.selected_pyramids) == 0:
            legend = {
                "A": "Enter add pyramid mode",
                "click and drag": "Select pyramids",
                "ESC": "Previous menu"
            }
        else:
            # When pyramids are selected, only show editing options
            legend = {
                "D": "Delete selected pyramids",
                "W": "Change selected leg",
                "E": "Increase leg length",
                "Q": "Decrease leg length",
                "Click anywhere": "Unselect pyramids",
                "ESC": "Previous menu"
            }
        return legend, ["pyramid_count", "selected_pyramid_count", ("Selected leg", "leg_index")]

    
    def add_pyramid(self, key=None, mousedrag=None):
        """Add a pyramid"""
        if mousedrag is not None:
            pos = mousedrag[0]
            length = max(np.linalg.norm(mousedrag[1] - mousedrag[0]), 4)
            self.IA.pyramids += [Pyramid(self.IA, pos, id=len(self.IA.pyramids))]
            self.IA.pyramids[-1].set_cross_lengths([length, length, length, length])
            self.selected_pyramids = [self.IA.pyramids[-1]]
        else:
            return {'Click and drag': 'Add pyramid'}, []

    def set_micrometer_scale(self, key=None, mousedrag=None):
        """Set the micrometer scale length"""
        if mousedrag is not None:
            self.micrometer_scale_line = mousedrag
            self.micrometer_scale_length = np.linalg.norm(self.micrometer_scale_line[1] - self.micrometer_scale_line[0])
        else:
            arrow_coords = self.micrometer_scale_line - np.array((self.IA.gray.shape[0], 0))
            titlebar = self.titlebar_original.copy()
            self.IA.titlebar = titlebar
            line(titlebar, arrow_coords[0], arrow_coords[1], (0, 200, 15), 2)
            self.IA.show()
            return {'Click and drag': 'set scale line', 'n': self.set_micrometer_scale_length, 'ESC': 'Previous menu'}, ['micrometer_scale_length', 'um_per_pixel']
      
    def set_micrometer_scale_length(self, key=None, mousedrag=None):
        """Set numerical value of micrometer scale"""
        if key is not None:
            if key == K_BACKSPACE:
                self.length_in_um = self.length_in_um[:-1]
            elif key == K_ENTER:
                self.micrometer_scale_length = float(self.length_in_um)
                self.length_in_um = ""
                self.mode_function = self.set_micrometer_scale
            else:
                self.length_in_um += chr(key)
        return {"Enter": "Submit", "Backspace": "Delete"}, ["length_in_um"]

    def um_per_pixel(self):
        pix = np.linalg.norm(self.micrometer_scale_line[1] - self.micrometer_scale_line[0])
        um = self.micrometer_scale_length
        return um / pix

    def pixels_to_um(self, pixels):
        return pixels * self.um_per_pixel()

    def set_direction(self, key=None, mousedrag=None):
        """Set pyramid leg direction"""
        if mousedrag is not None:
            self.direction_vector = np.array(mousedrag[1]) - np.array(mousedrag[0])
            self.IA.rotation = np.arctan2(self.direction_vector[1], self.direction_vector[0])
            self.direction_vector_arrow = mousedrag
        
        # Create display with direction vector arrow
        display_img = self.IA.img.copy()
        
        # Draw the current direction vector arrow if it exists
        if hasattr(self, 'direction_vector_arrow') and len(self.direction_vector_arrow) == 2:
            start_point = self.direction_vector_arrow[0]
            end_point = self.direction_vector_arrow[1]
            # Draw a thick blue arrow
            arrow(display_img, start_point, end_point, (0, 0, 255), 3)
        
        self.guicanvas = display_img
        
        return {
            "Click and drag": "Set direction vector", 
            "ESC": "Previous menu"
        }, [("Rotation (radians)", lambda: f"{self.IA.rotation:.3f}")]

    def computations_menu(self, key=None, mousedrag=None):
        """Perform computations"""
        if key == ord('S') or key == ord('s'):
            self.IA.segment(20)
            return self.computations_menu()
        elif key == ord('L') or key == ord('l'):
            self.IA.get_leg_lengths()
            return self.computations_menu()
        elif key == ord('A') or key == ord('a'):
            self.IA.consolidate_pyramids()
            return self.computations_menu()
        elif key == ord('H') or key == ord('h'):
            return self.make_histogram()
        
        legend = {
            "S": "Detect pyramids",
            "L": "Calculate leg lengths",
            "A": "Absorb pyramids that overlap",
            "H": "Make a histogram of the pyramid heights",
            "ESC": "Previous menu"
        }
        return legend, [("Pyramids", "pyramid_count")]

    def consolidate_pyramids_menu(self, key=None, mousedrag=None):
        """Consolidate overlapping pyramids"""
        if key is not None or mousedrag is not None:
            return None
        print("Consolidating pyramids...")
        self.IA.consolidate_pyramids()
        print(f"Now have {len(self.IA.pyramids)} pyramids")
        return self.computations_menu()

    def calculate_leg_lengths(self, key=None, mousedrag=None):
        """Calculate leg lengths"""
        if key is not None or mousedrag is not None:
            return None
        print("Calculating leg lengths...")
        self.IA.get_leg_lengths()
        print("Done!")
        return self.computations_menu()

    def make_histogram(self, key=None, mousedrag=None):
        """Make histogram of pyramid heights"""
        if key is not None or mousedrag is not None:
            return None
        
        # Prompt user for save location
        from tkinter.filedialog import asksaveasfilename
        import tkinter as tk
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Suggest default filename based on image name
        default_name = self.IA.imgstr.split("/")[-1].split("\\")[-1].split(".")[0] + "_histogram.png"
        
        file_path = asksaveasfilename(
            title="Save Histogram",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        
        if not file_path:
            print("Histogram save cancelled")
            return self.computations_menu()
        
        print(f"Creating histogram...")
        
        # Generate histogram
        heights, work, heights_weighted, Base_areas = height_calc(self, self.IA.pyramids)
        titles = ["Maximum Height", "Minimum Height", "Average Height"]
        paths = []
        for i in range(len(titles)):
            paths.append(make_hist_with_subplots(self, heights[i], titles[i], work[i], 
                                                 heights_weighted[i], Base_areas[i], subplot_params=False))
        
        # Combine images
        from PIL import Image
        images = list(map(Image.open, paths))
        widths_hist, heights_hist = zip(*(i.size for i in images))
        
        # Get original image (without pyramids drawn) and processed image (with pyramids)
        orig_img_array = np.concatenate((self.IA.img, self.IA.titlebar), axis=0)
        processed_img_array = np.concatenate((self.IA.last_show, self.IA.titlebar), axis=0)
        
        # Convert to PIL for easier manipulation
        orig_img = Image.fromarray(orig_img_array)
        processed_img = Image.fromarray(processed_img_array)
        
        # Calculate total histogram width
        total_histogram_width = sum(widths_hist)
        max_histogram_height = max(*heights_hist)
        
        # Add small gap between top images (1/100th of total width)
        gap = total_histogram_width // 100
        
        # Each top image width: (total_width - gap) / 2
        target_img_width = (total_histogram_width - gap) // 2
        
        # Resize images to match width (maintaining aspect ratio)
        orig_aspect = orig_img.height / orig_img.width
        new_img_height = int(target_img_width * orig_aspect)
        
        orig_img_resized = orig_img.resize((target_img_width, new_img_height), Image.LANCZOS)
        processed_img_resized = processed_img.resize((target_img_width, new_img_height), Image.LANCZOS)
        
        # Create final image - total width matches histogram width
        total_width = total_histogram_width
        total_height = new_img_height + max_histogram_height
        
        new_im = Image.new("RGB", (total_width, total_height), color='white')
        
        # Paste original image on the left
        new_im.paste(orig_img_resized, (0, 0))
        
        # Paste processed image on the right (with gap)
        new_im.paste(processed_img_resized, (target_img_width + gap, 0))
        
        # Paste histograms below
        y_offset = new_img_height
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
        
        # Save to user-specified location
        new_im.save(file_path)
        print(f"Histogram saved to: {file_path}")
        
        # Also save text file and Excel in the same directory
        imgname = self.IA.imgstr.split("/")[-1].split("\\")[-1].split(".")[0]
        output_dir = os.path.dirname(file_path)
        
        # Create Excel file with all data
        dataframe = pd.DataFrame({
            "Max Height (um)": heights[0], 
            "Max Height Calculations": work[0], 
            "Min Height (um)": heights[1], 
            "Min Height Calculations": work[1], 
            "Avg Height (um)": heights[2], 
            "Avg Height Calculations": work[2]
        })
        excel_path = os.path.join(output_dir, f"{imgname}_heights.xlsx")
        dataframe.to_excel(excel_path, index=False)
        print(f"Excel data saved to: {excel_path}")
        
        # Save single comprehensive text file with all height data
        txt_path = os.path.join(output_dir, f"{imgname}_heights.txt")
        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Pyramid Height Analysis for: {imgname}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write statistics for each measure
            for i, title in enumerate(titles):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{title.upper()}\n")
                f.write(f"{'=' * 80}\n\n")
                
                # Summary statistics
                f.write(f"Mean: {np.mean(heights[i]):.3f} um\n")
                f.write(f"Area-Weighted Mean: {np.sum(heights_weighted[i])/np.sum(Base_areas[i]):.3f} um\n")
                f.write(f"Median: {np.median(heights[i]):.3f} um\n")
                f.write(f"Std Dev: {np.std(heights[i]):.3f} um\n")
                f.write(f"Min: {np.min(heights[i]):.3f} um\n")
                f.write(f"Max: {np.max(heights[i]):.3f} um\n")
                f.write(f"Total Pyramids: {len(heights[i])}\n\n")
                
                # Individual pyramid data
                f.write(f"Individual Pyramid Data:\n")
                f.write(f"{'-' * 80}\n")
                for j in range(len(heights[i])):
                    f.write(f"Pyramid {j+1}: {heights[i][j]:.3f} um\n")
                    f.write(f"  Calculation: {work[i][j]}\n\n")
        
        print(f"Text file saved to: {txt_path}")
        
        return self.computations_menu()

# ============================================================================
# File Selection Utility
# ============================================================================

def select_image_file():
    """Open a file dialog to select an image file"""
    print("Opening file dialog...")
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    file_path = askopenfilename(
        title="Select Image File",
        filetypes=[
            ("TIFF files", "*.tif *.tiff"),
            ("All image files", "*.tif *.tiff *.png *.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    
    if not file_path:
        print("No file selected. Exiting...")
        return None
    
    print(f"File selected: {file_path}")
    return file_path


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function for pyramid analysis with GUI control interface.
    """
    print("Starting pyramid analysis GUI...")
    print("\n=== Controls ===")
    print("Press 'S' - Setup menu (segment, set scale, set direction)")
    print("Press 'P' - Pyramid edit mode (select and edit pyramids)")
    print("Press 'A' - Add pyramid")
    print("Press 'C' - Computations menu (segment, leg lengths, consolidate, histogram)")
    print("Press 'ESC' - Return to main menu")
    print("Press 'Q' - Quit")
    print("\n")
    
    gui = GuiControl()
    if gui.IA is not None:
        print(f"Image loaded successfully!")
        
        # Just show the window normally - no topmost tricks
        gui.IA.show(gui.IA.img, ms=1)
        
        print("\n" + "="*60)
        print(">>> CLICK ON THE 'Image Analysis' WINDOW TO START <<<")
        print("="*60 + "\n")
        print("Starting GUI control loop...\n")
        gui.control_loop()
    else:
        print("No image loaded. Exiting...")
    
if __name__ == '__main__':
    main()
    print("Done!")