# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:54:45 2025

@author: 37067
"""
# DISCLAIMER: These are not meant to be universal functions suitable for any code.
# Implementing in other codes may require significant changes.

import numpy as np
from scipy import ndimage as ndim
import polars as pl
from polars import col as c

def normalize_to_1(smoothened_array):
    """
    normalizes the array to, making smallest value 0 and larges value 1.
    :param cv_array: np.array of covariance values
    :return: normalized array
    """

    try:
        denoised_array = (smoothened_array - min(smoothened_array))
        normalized_array = denoised_array / max(denoised_array)
        return normalized_array
    except ZeroDivisionError:
        return smoothened_array

def find_inflection_points(scaled_array):
    """
    calculates the inflection points by approximating where 2nd
    derivative of the array crosses 0.

    :param cv_array: np.array of covariances
    :return: np.arrays of inflection points
    """
    smoothd2 = ndim.gaussian_filter1d(np.gradient(np.gradient(scaled_array)), 2)
    inflection_points = np.where(np.diff(np.sign(smoothd2))) [0]
    return inflection_points

def filter_inflection_points(peaks, troughs, inflection_points, starvation_onset, time_of_interest=120, img_rate=3):
    """peaks, troughs, inflection_points are np.arrays, starvation_onset, time_of_interest, img_rate are ints.
    default params: time_of_interest=120, img_rate=3
    filters troughs that come before the first peak,
    then filters inflection points that are between peaks and troughs,
    then filters inflection points for the time span between time of interest (default=120) and starvation onset,
    then finds the time before starvation the inflection point occured.
    returns the final inflection points and their difference with starvation onset."""
    if len(peaks) > 0:
        #removes troughs that come before the first peak.
        trough_start_value = min(peaks)
        troughs_filtered=[trough for trough in troughs if trough_start_value < trough]

    infls_filtered=[]
    for i in range(len(peaks)-1):
        #find inflection points between each peak and trough
        infls_filtered.extend( [ p for p in inflection_points if peaks[i] < p < troughs_filtered[i] ] )

    infls_filtered2 = infls_filtered[:]
    # print(infls_filtered2)
    for p in infls_filtered:
        # remove inflection points if they're not within time of interest before starvation
        if p > starvation_onset / img_rate or starvation_onset / img_rate - p > time_of_interest / img_rate:
           infls_filtered2.remove(p)
    # print(infls_filtered2)

    # find time before starvation
    infls_difference=[starvation_onset / img_rate - p for p in infls_filtered2]
    return infls_filtered2, infls_difference

def find_change_point(y_values_starvation,slope_values_starvation,points_after,factor):
    """y_values_starvation, slope_values_starvation are lists, points_after is an int, factor is float<1
    first selects the slope values from indices i:i+points_after
    then if they rise slower (1+factor/slope_value[i]) per point,
    selects slope index i to be point_low_x,
    and slope value i to be point_low_y.
    returns both selected points
    returns None if such conditions are not met throughout the list of slope_values"""
    if len(y_values_starvation)>1:
        for idx, y in enumerate(y_values_starvation): 
            #select the slope values
            slopes_interim = slope_values_starvation[idx:idx+points_after]
            if (slopes_interim[-1]/slopes_interim[0])>((1+factor/slopes_interim[0])**points_after) and all(s>0 for s in slopes_interim):
                #selects x and y points
                point_low_x = idx
                point_low_y = y_values_starvation[idx]
                # print(f"low point x,y: {point_low_x, point_low_y}")
                break
                #print(point_low_x)
                #print(slopes_interim)
            else:
                point_low_x, point_low_y=None,None
        return point_low_x, point_low_y
    else:
        point_low_x, point_low_y=None,None
        return point_low_x,point_low_y
    
def find_high_point(y_values_starvation,factor=0.8):
    """y_values_starvation is a list, factor is a float<1.
    Default params: float=0.8
    if y is larger than maximum y_value in list multiplied by factor,
    point_high_x becomes ys idx
    point_low_x becomes y
    returns point_high_x and point_high_y
    returns them as None if no such conditions are met throughout the list of y_values"""
    if len(y_values_starvation)>1:
        for idx, y in enumerate(y_values_starvation):
            # print(f"maximum: {max(y_values_starvation)}")
            if y>factor*( max(y_values_starvation) ):
                # print(f"80% of max: {0.8*max(y_values_starvation)}")
                point_high_x = idx #returns index of points where both conditions are true for the first time
                point_high_y = y
                #print(point_high_x, point_high_y)
                return point_high_x, point_high_y
            else:
                point_low_x, point_low_y = None,None
    else:
        point_low_x, point_low_y=None,None
        return point_low_x,point_low_y

def prepare_df(path_to_df):
    # retrieves a pl.Series of unique Cell IDs, converts to np.array
    # and flattens (pl.Series cannot be iterated over; non-flattened pl.Series are n-dimensional)
    full_df = pl.read_csv(path_to_df)
    unique_IDs = (
        full_df.unique(subset=["Cell_ID"]).select(c("Cell_ID")).to_numpy().flatten()
    )
    unique_IDs.sort()
    return full_df, unique_IDs

def extract_x_and_y(cell_df, channel_1, channel_2):
    x = cell_df["time_minutes"]
    if channel_2 is not "":
        y_signal_1 = cell_df[channel_1]
        y_signal_2 = cell_df[channel_2]
        y = y_signal_1 / y_signal_2
    else:
        y = cell_df[channel_1]  # e.g. Quad2_mCherry_CV
    return x, y
