#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import torch
import logging
from math import radians, sin, cos, asin, sqrt

# ==============================================================================
# 1. Reference Coordinates for NHS Regions
# ==============================================================================
REFERENCE_COORDINATES = {
    "East of England": (52.1766, 0.425889),
    "Midlands": (52.7269, -1.458210),
    "London": (51.4923, -0.308660),
    "South East": (51.4341, -0.969570),
    "South West": (50.8112, -3.633430),
    "North West": (53.8981, -2.657550),
    "North East and Yorkshire": (54.5378, -2.180390),
}

# ==============================================================================
# 2. Data Loading and Preprocessing Functions
# ==============================================================================
def load_and_correct_data(data: pd.DataFrame,
                          reference_coordinates: dict) -> pd.DataFrame:
    """
    Assigns correct latitude and longitude for each region based on the 
    provided reference coordinates. Then applies a 7-day rolling mean to 
    selected daily features and sorts the data chronologically.
    
    Parameters:
        data (pd.DataFrame): Raw data.
        reference_coordinates (dict): Dictionary mapping region names to (lat, lon).
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Overwrite lat/lng with reference coordinates for each region
    for region, coords in reference_coordinates.items():
        data.loc[data["areaName"] == region, ["latitude", "longitude"]] = coords

    # Define features to smooth with a 7-day rolling mean
    rolling_features = ['new_confirmed', 'new_deceased', 'newAdmissions',
                        'hospitalCases', 'covidOccupiedMVBeds']

    data[rolling_features] = (
        data.groupby('areaName')[rolling_features]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )

    # Fill any remaining missing values with 0
    data[rolling_features] = data[rolling_features].fillna(0)

    # Sort data by region and date (chronologically)
    data.sort_values(["areaName", "date"], inplace=True)
    return data

def create_target_timeseries(data: pd.DataFrame,
                             target_feature: str = "covidOccupiedMVBeds") -> pd.DataFrame:
    """
    Pivots the data so that rows are dates and columns are regions,
    using the target feature.
    
    Parameters:
        data (pd.DataFrame): Processed DataFrame.
        target_feature (str): Feature to use as the target.
    
    Returns:
        pd.DataFrame: Pivoted time-series DataFrame.
    """
    pivot = data.pivot(index='date', columns='areaName', values=target_feature)
    pivot.ffill(inplace=True)
    pivot.fillna(0, inplace=True)
    return pivot

# ==============================================================================
# 3. Geographic Adjacency (Similarity) Functions
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points on Earth.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth’s radius in kilometers
    return c * r

def compute_geographic_adjacency(regions: list,
                                 latitudes: list,
                                 longitudes: list,
                                 threshold: float = 150) -> torch.Tensor:
    """
    Creates a binary adjacency matrix based on geographic proximity.
    If the Haversine distance between two regions is less than or equal to
    the threshold (in km), an edge is set (value 1), otherwise 0.
    
    Parameters:
        regions (list): List of region names.
        latitudes (list): List of latitudes corresponding to regions.
        longitudes (list): List of longitudes corresponding to regions.
        threshold (float): Distance threshold in km.
    
    Returns:
        torch.Tensor: Adjacency matrix (num_regions x num_regions).
    """
    num_nodes = len(regions)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                adj_matrix[i][j] = 1  # self-loop
            else:
                distance = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                if distance <= threshold:
                    adj_matrix[i][j] = 1
                    # Ensure symmetry:
                    adj_matrix[j][i] = 1
    return torch.tensor(adj_matrix, dtype=torch.float32)

# ==============================================================================
# 4. Main Preprocessing Routine
# ==============================================================================
def main():
    logging.basicConfig(level=logging.INFO)
    
    # Construct paths relative to this script's directory
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    input_csv = os.path.join(base_dir, "data", "merged_nhs_covid_data.csv")
    
    if not os.path.exists(input_csv):
        logging.error(f"Input CSV file not found: {input_csv}")
        return

    logging.info(f"Loading raw data from {input_csv}...")
    # Parse the 'date' column as dates
    raw_data = pd.read_csv(input_csv, parse_dates=["date"])

    # Process the data (assign coordinates, apply 7-day rolling mean, sort)
    data_processed = load_and_correct_data(raw_data, REFERENCE_COORDINATES)
    logging.info("Data processed with 7-day rolling mean and sorted.")

    # Create pivot table for the target feature (covidOccupiedMVBeds)
    pivot = create_target_timeseries(data_processed, target_feature="covidOccupiedMVBeds")
    logging.info("Pivot table for target variable created.")

    # Save the time-series matrix as a txt file (without headers or index)
    output_timeseries = os.path.join(base_dir, "data", "nhs_timeseries.txt")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_timeseries), exist_ok=True)
    pivot.to_csv(output_timeseries, header=False, index=False)
    logging.info(f"Saved processed time-series data to {output_timeseries}")

    # Compute the geographic adjacency matrix based on the corrected coordinates.
    region_coords = data_processed.groupby("areaName").first()[["latitude", "longitude"]]
    regions = list(region_coords.index)
    latitudes = region_coords["latitude"].tolist()
    longitudes = region_coords["longitude"].tolist()
    threshold_distance = 150  # km
    adj_matrix = compute_geographic_adjacency(regions, latitudes, longitudes, threshold=threshold_distance)
    
    # Save the adjacency matrix as a comma-separated txt file.
    output_adj = os.path.join(base_dir, "data", "nhs-adj.txt")
    os.makedirs(os.path.dirname(output_adj), exist_ok=True)
    np.savetxt(output_adj, adj_matrix.numpy(), fmt="%.0f", delimiter=",")
    logging.info(f"Saved geographic adjacency matrix to {output_adj}")

if __name__ == "__main__":
    main()
