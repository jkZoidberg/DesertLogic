import rioxarray
import matplotlib.pyplot as plt

def visualize_oasis_mask(tif_file_path):
    # Load the GeoTIFF using rioxarray
    da = rioxarray.open_rasterio(tif_file_path)
    
    # Plot the raster data
    plt.figure(figsize=(10, 10))
    img = da.plot(cmap='viridis')  # Store the plot image object

    # Add a colorbar using the mappable object 'img'
    plt.colorbar(img, label='Oasis Mask Value')  # Pass the 'img' object to colorbar
    plt.title("Oasis Mask")
    plt.show()

# Example usage
tif_file_path = "out/oasis_mask_202507.tif"  # Replace with your actual file path
visualize_oasis_mask(tif_file_path)