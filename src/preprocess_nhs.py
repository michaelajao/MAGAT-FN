"""
preprocess_nhs.py

This script loads the raw NHS COVID data, corrects the geographic coordinates 
(using reference coordinates), applies a 7-day rolling mean to smooth daily 
features, sorts the data, creates a pivot table for the target variable, and 
computes the geographic adjacency matrix. Finally, it saves the time-series data, 
the adjacency matrix, and a visualization of the matrix.
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import logging
from math import radians, sin, cos, asin, sqrt
import matplotlib.pyplot as plt
import seaborn as sns

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
    For each region, overwrites the latitude and longitude using the given
    reference coordinates, applies a 7-day rolling mean to the selected daily 
    features, fills any missing values with 0, and sorts the data chronologically.
    """
    # Overwrite lat/lng for each region
    for region, coords in reference_coordinates.items():
        data.loc[data["areaName"] == region, ["latitude", "longitude"]] = coords

    # Define daily features to smooth
    rolling_features = ['new_confirmed', 'new_deceased', 'newAdmissions',
                        'hospitalCases', 'covidOccupiedMVBeds']

    # Apply 7-day rolling mean per region
    data[rolling_features] = (
        data.groupby('areaName')[rolling_features]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )

    # Fill missing values with 0 and sort by region and date
    data[rolling_features] = data[rolling_features].fillna(0)
    data.sort_values(["areaName", "date"], inplace=True)
    return data

def create_target_timeseries(data: pd.DataFrame,
                             target_feature: str = "covidOccupiedMVBeds") -> pd.DataFrame:
    """
    Pivots the data so that rows are dates and columns are regions,
    using the specified target feature.
    """
    pivot = data.pivot(index='date', columns='areaName', values=target_feature)
    pivot.ffill(inplace=True)
    pivot.fillna(0, inplace=True)
    return pivot

# ==============================================================================
# 3. Geographic Adjacency Functions
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points on Earth.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth’s radius in kilometers
    return c * r

def compute_geographic_adjacency(regions: list,
                                 latitudes: list,
                                 longitudes: list,
                                 threshold: float = 150) -> torch.Tensor:
    """
    Creates a binary adjacency matrix using the Haversine formula. Two regions
    are connected (edge=1) if the distance between them is less than or equal to
    the threshold.
    """
    num_nodes = len(regions)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                adj_matrix[i][j] = 1  # Self-loop
            else:
                distance = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                if distance <= threshold:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1  # Ensure symmetry
    return torch.tensor(adj_matrix, dtype=torch.float32)

# ==============================================================================
# 4. Main Preprocessing Routine
# ==============================================================================
def main():
    logging.basicConfig(level=logging.INFO)
    # Determine the base directory (one level up from src)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")

    input_csv = os.path.join(data_dir, "merged_nhs_covid_data.csv")
    if not os.path.exists(input_csv):
        logging.error(f"Input CSV file not found: {input_csv}")
        return

    logging.info(f"Loading raw data from {input_csv}...")
    raw_data = pd.read_csv(input_csv, parse_dates=["date"])

    # Process data: correct coordinates, smooth using rolling mean, and sort.
    data_processed = load_and_correct_data(raw_data, REFERENCE_COORDINATES)
    logging.info("Data processed with 7-day rolling mean and sorted.")

    # Create pivot table for target variable.
    pivot = create_target_timeseries(data_processed, target_feature="covidOccupiedMVBeds")
    logging.info("Pivot table for target variable created.")

    # Save the time-series data into a text file.
    output_timeseries = os.path.join(data_dir, "nhs_timeseries.txt")
    pivot.to_csv(output_timeseries, header=False, index=False)
    logging.info(f"Saved processed time-series data to {output_timeseries}")

    # Compute the geographic adjacency matrix using reference coordinates.
    region_coords = data_processed.groupby("areaName").first()[["latitude", "longitude"]]
    regions = list(region_coords.index)
    latitudes = region_coords["latitude"].tolist()
    longitudes = region_coords["longitude"].tolist()
    threshold_distance = 150  # Threshold distance in km
    adj_matrix = compute_geographic_adjacency(regions, latitudes, longitudes, threshold=threshold_distance)

    # Save the adjacency matrix to a text file.
    output_adj = os.path.join(data_dir, "nhs-adj.txt")
    np.savetxt(output_adj, adj_matrix.numpy(), fmt="%.0f", delimiter=",")
    logging.info(f"Saved geographic adjacency matrix to {output_adj}")

    # Plot and save the adjacency matrix as a heatmap.
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(adj_matrix.numpy(), cmap="viridis", annot=True, fmt=".0f", cbar=True, ax=ax)
    ax.set_title("Geographic Adjacency Matrix")
    ax.set_xlabel("Region Index")
    ax.set_ylabel("Region Index")
    output_adj_img = os.path.join(data_dir, "nhs-adj.png")
    plt.savefig(output_adj_img)
    plt.close(fig)
    logging.info(f"Saved adjacency matrix visualization to {output_adj_img}")

if __name__ == "__main__":
    main()
