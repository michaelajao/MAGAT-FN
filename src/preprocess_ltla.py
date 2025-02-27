"""
preprocess_ltla.py

This script loads the raw LTLA COVID data from merged_LTLA_data.csv,
converts the population column to a numeric type, applies a 7-day rolling mean 
to the daily cases, sorts the data by area and date, creates a pivot table 
for the target variable, and computes the geographic adjacency matrix using 
the provided latitude and longitude. Finally, it saves the time-series data, 
the adjacency matrix, and a visualization of the matrix.
"""

import os
import numpy as np
import pandas as pd
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, asin, sqrt

def load_and_preprocess_ltla(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the LTLA data:
      - Converts the population column from string (with commas) to integer.
      - Parses the date column.
      - Sorts the data by areaName and date.
      - Applies a 7-day rolling mean to the dailyCases.
    """
    # Convert population to integer (remove commas)
    data['population'] = data['population'].replace({',': ''}, regex=True).astype(int)
    
    # Ensure the date column is parsed as datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort data by areaName and date
    data.sort_values(["areaName", "date"], inplace=True)
    
    # Apply a 7-day rolling mean on dailyCases per areaName
    data['dailyCases_smoothed'] = data.groupby('areaName')['dailyCases'] \
                                      .rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    
    return data

def create_target_timeseries(data: pd.DataFrame, target_feature: str = "dailyCases_smoothed") -> pd.DataFrame:
    """
    Creates a pivot table with dates as rows and area names as columns,
    using the specified target feature.
    """
    pivot = data.pivot(index='date', columns='areaName', values=target_feature)
    pivot.ffill(inplace=True)
    pivot.fillna(0, inplace=True)
    return pivot

def haversine(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points on Earth.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth's radius in kilometers
    return c * r

def compute_geographic_adjacency(regions: list, latitudes: list, longitudes: list, threshold: float = 150) -> torch.Tensor:
    """
    Computes a binary geographic adjacency matrix using the Haversine formula.
    Two areas are connected (edge=1) if the distance between them is less than or equal to the threshold.
    """
    num_nodes = len(regions)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i, num_nodes):  # Use symmetry: only compute for j >= i
            if i == j:
                adj_matrix[i][j] = 1  # Self-loop
            else:
                distance = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                if distance <= threshold:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1  # Ensure symmetry
    return torch.tensor(adj_matrix, dtype=torch.float32)

def main():
    logging.basicConfig(level=logging.INFO)
    # Determine base directory (assumes this script is in src/ and data/ is one level up)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    
    input_csv = os.path.join(data_dir, "updated_LTLA_fixed.csv")
    if not os.path.exists(input_csv):
        logging.error(f"Input CSV file not found: {input_csv}")
        return
    
    logging.info(f"Loading raw data from {input_csv}...")
    raw_data = pd.read_csv(input_csv)
    
    # Preprocess data: convert population, parse date, sort, and apply rolling mean
    data_processed = load_and_preprocess_ltla(raw_data)
    logging.info("Data preprocessed: population converted, rolling mean applied, sorted by area and date.")
    
    # Create pivot table for the target variable (smoothed daily cases)
    pivot = create_target_timeseries(data_processed, target_feature="dailyCases_smoothed")
    logging.info("Pivot table for target variable created.")
    
    # Save the pivot table (time-series data) to a text file
    output_timeseries = os.path.join(data_dir, "ltla_timeseries.txt")
    pivot.to_csv(output_timeseries, header=False, index=False)
    logging.info(f"Saved processed time-series data to {output_timeseries}")
    
    # Compute the geographic adjacency matrix using the provided lat/long.
    # We group by areaName and take the first occurrence to get the coordinates for each area.
    area_coords = data_processed.groupby("areaName").first()[["lat", "long"]]
    regions = list(area_coords.index)
    latitudes = area_coords["lat"].tolist()
    longitudes = area_coords["long"].tolist()
    threshold_distance = 150  # Threshold distance in km (adjust if needed)
    adj_matrix = compute_geographic_adjacency(regions, latitudes, longitudes, threshold=threshold_distance)
    
    # Save the adjacency matrix to a text file
    output_adj = os.path.join(data_dir, "ltla-adj.txt")
    np.savetxt(output_adj, adj_matrix.numpy(), fmt="%.0f", delimiter=",")
    logging.info(f"Saved geographic adjacency matrix to {output_adj}")
    
    # Plot and save the adjacency matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(adj_matrix.numpy(), cmap="viridis", annot=True, fmt=".0f", cbar=True, ax=ax)
    ax.set_title("Geographic Adjacency Matrix (LTLA)")
    ax.set_xlabel("Area Index")
    ax.set_ylabel("Area Index")
    output_adj_img = os.path.join(data_dir, "ltla-adj.png")
    plt.savefig(output_adj_img)
    plt.close(fig)
    logging.info(f"Saved adjacency matrix visualization to {output_adj_img}")

if __name__ == "__main__":
    main()
