# ================================================================
# Terrain Generator with Unified Roughness Scaling
# Author: Xiyuan Wang + GPT-5 + Claude (2025)
# ================================================================
# This script generates terrain heightmaps where roughness controls
# BOTH the noise amplitude AND the pit/gap characteristics.
# It also creates organized folder structures for MuJoCo simulation.
# ================================================================

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import os
import shutil

# ================================================================
# === Configuration ===
# ================================================================

# Mesh grid setup
SIZE = 1024
x = np.linspace(0, 4 * np.pi, SIZE)
y = np.linspace(0, 4 * np.pi, SIZE)
X, Y = np.meshgrid(x, y)

# Template folder path (modify this to match your system)
TEMPLATE_FOLDER = "/home/xqw5351/farms_ws/farms_centipede_experiments-master/models/1-Arena_maker/2-Continuous_terrain/terrain_test/-1"

# Output base directory (where new folders will be created)
OUTPUT_BASE_DIR = "/home/xqw5351/farms_ws/farms_centipede_experiments-master/models/1-Arena_maker/2-Continuous_terrain/terrain_test"

# ================================================================
# === Unified Roughness Scaling ===
# ================================================================

def get_feature_params(roughness):
    """
    Scale pit/gap parameters with terrain roughness.
    
    Design rationale:
    - Rougher terrain has MORE pits (increased geological activity)
    - Rougher terrain has DEEPER pits/gaps (more extreme features)
    - Rougher terrain has SMALLER individual pits (many small irregular features)
    - Rougher terrain spreads hazards more uniformly (higher concentration_sigma)
    
    Args:
        roughness: float from 0.0 to 2.0
        
    Returns:
        dict with all pit/gap parameters scaled to roughness
    """
    if roughness <= 0.0:
        # Flat terrain - no pits or gaps
        return {
            'num_pits': 0,
            'num_gaps': 0,
            'pit_depth': 0,
            'gap_depth': 0,
            'pit_radius_range': (3, 5),
            'gap_width_range': (2, 3),
            'concentration_sigma': 0.05,
        }
    
    # Scale parameters with roughness
    return {
        'num_pits': int(20 + roughness * 80),                    # 20–180
        'num_gaps': int(5 + roughness * 15),                     # 5–35
        'pit_depth': 5 + roughness * 10,                         # 5–25
        'gap_depth': 5 + roughness * 12,                         # 5–29
        'pit_radius_range': (
            max(3, int(8 - roughness * 2)),                      # 8→4 (smaller with roughness)
            max(5, int(12 - roughness * 3))                      # 12→6
        ),
        'gap_width_range': (
            max(2, int(5 - roughness)),                          # 5→3
            max(3, int(8 - roughness * 2))                       # 8→4
        ),
        'concentration_sigma': 0.1,          # 0.05–0.35 (spread out)
    }


def print_feature_params(roughness):
    """Print the feature parameters for a given roughness level."""
    params = get_feature_params(roughness)
    print(f"\n  Roughness {roughness:.1f} parameters:")
    print(f"    num_pits: {params['num_pits']}")
    print(f"    num_gaps: {params['num_gaps']}")
    print(f"    pit_depth: {params['pit_depth']:.1f}")
    print(f"    gap_depth: {params['gap_depth']:.1f}")
    print(f"    pit_radius_range: {params['pit_radius_range']}")
    print(f"    gap_width_range: {params['gap_width_range']}")
    print(f"    concentration_sigma: {params['concentration_sigma']:.2f}")


# ================================================================
# === Utility Filters ===
# ================================================================

def apply_smoothing_filter(noise, filter_type='gaussian', sigma=1.0):
    """Apply smoothing filter to noise to eliminate sharp edges."""
    if filter_type == 'gaussian':
        return gaussian_filter(noise, sigma=sigma)
    elif filter_type == 'uniform':
        kernel_size = int(sigma * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        return ndimage.convolve(noise, kernel)
    elif filter_type == 'median':
        return ndimage.median_filter(noise, size=int(sigma * 2 + 1))
    else:
        return noise


# ================================================================
# === Perlin-like Noise Generator ===
# ================================================================

def generate_perlin_like_noise(size, scale=50, octaves=4, persistence=0.5):
    """Generate Perlin-like noise for more natural terrain."""
    noise = np.zeros((size, size))
    frequency = 1.0
    amplitude = 1.0
    max_value = 0.0
    
    for i in range(octaves):
        x_freq = np.arange(size) * frequency / scale
        y_freq = np.arange(size) * frequency / scale
        
        gradient_noise = np.random.randn(
            int(size * frequency / scale) + 1,
            int(size * frequency / scale) + 1
        )
        resized_noise = ndimage.zoom(gradient_noise, scale / frequency, order=1)
        
        if resized_noise.shape[0] > size:
            resized_noise = resized_noise[:size, :size]
        elif resized_noise.shape[0] < size:
            pad_width = (
                (0, size - resized_noise.shape[0]),
                (0, size - resized_noise.shape[1])
            )
            resized_noise = np.pad(resized_noise, pad_width, mode='edge')
            
        noise += resized_noise * amplitude
        max_value += amplitude
        
        amplitude *= persistence
        frequency *= 2
    
    return noise / max_value


# ================================================================
# === Pits and Gaps Generator ===
# ================================================================

def add_pits_and_gaps(heightmap, roughness, random_seed=42):
    """
    Add circular pits and linear gaps scaled by roughness.
    
    Args:
        heightmap: 2D numpy array of terrain heights
        roughness: float controlling feature intensity (0.0–2.0)
        random_seed: for reproducibility
        
    Returns:
        Modified heightmap with pits and gaps
    """
    params = get_feature_params(roughness)
    
    if params['num_pits'] == 0 and params['num_gaps'] == 0:
        return heightmap
    
    rng = np.random.default_rng(random_seed)
    modified = heightmap.copy()
    size = heightmap.shape[0]
    
    center_x = size / 2
    center_y = size / 2
    sigma = params['concentration_sigma']
    
    def sample_centered_position():
        """Draw pit/gap centers from Gaussian centered on map center."""
        cx = rng.normal(center_x, size * sigma)
        cy = rng.normal(center_y, size * sigma)
        cx = int(np.clip(cx, 0, size - 1))
        cy = int(np.clip(cy, 0, size - 1))
        return cx, cy
    
    # --- Add pits (Gaussian depressions) ---
    for _ in range(params['num_pits']):
        cx, cy = sample_centered_position()
        radius = rng.integers(
            params['pit_radius_range'][0],
            params['pit_radius_range'][1] + 1
        )
        y_grid, x_grid = np.ogrid[:size, :size]
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        pit = np.exp(-0.5 * (dist / (radius / 2))**2)
        pit = (pit / pit.max()) * (-params['pit_depth'])
        modified += pit
    
    # --- Add gaps (linear trenches) ---
    for _ in range(params['num_gaps']):
        orientation = rng.uniform(0, np.pi)
        width = rng.integers(
            params['gap_width_range'][0],
            params['gap_width_range'][1] + 1
        )
        x0, y0 = sample_centered_position()
        x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
        dist = np.abs(
            (x_grid - x0) * np.cos(orientation) +
            (y_grid - y0) * np.sin(orientation)
        )
        trench = np.exp(-0.5 * (dist / (width / 2))**2)
        trench = (trench / trench.max()) * (-params['gap_depth'])
        modified += trench
    
    # Blend features smoothly
    modified = gaussian_filter(modified, sigma=2.0)
    return modified


# ================================================================
# === Terrain Generation ===
# ================================================================

def generate_terrain(roughness, smoothing='gaussian', use_perlin=True, random_seed=42):
    """
    Generate terrain with unified roughness scaling.
    
    Args:
        roughness: float from 0.0 to 2.0 controlling overall terrain difficulty
        smoothing: filter type ('gaussian', 'uniform', 'median')
        use_perlin: whether to use Perlin-like noise
        random_seed: for reproducibility
        
    Returns:
        heightmap as uint8 numpy array (0–255)
    """
    # Base terrain (gentle sine waves)
    base_terrain = ((np.sin(X / 2) * np.cos(Y / 2) + 1) * 50 + 128)
    
    # Add noise scaled by roughness
    if use_perlin:
        np.random.seed(random_seed)
        noise = generate_perlin_like_noise(SIZE, scale=100, octaves=4)
        noise = noise * roughness * 40
    else:
        np.random.seed(random_seed)
        noise = np.random.randn(SIZE, SIZE) * roughness * 30
    
    # Smooth the noise
    smoothing_sigma = max(0.5, roughness * 1.5)
    noise = apply_smoothing_filter(noise, smoothing, sigma=smoothing_sigma)
    
    heightmap = base_terrain + noise
    
    # Add pits and gaps (scaled by roughness)
    if roughness > 0:
        heightmap = add_pits_and_gaps(heightmap, roughness, random_seed=random_seed)
    
    # Final smoothing and clipping
    heightmap = gaussian_filter(heightmap, sigma=0.3)
    heightmap = np.clip(heightmap, 0, 255).astype(np.uint8)
    
    return heightmap


# ================================================================
# === Folder Management ===
# ================================================================

def setup_terrain_folder(roughness, heightmap, template_folder, output_base_dir):
    """
    Create a terrain folder by copying template and adding the heightmap.
    
    Args:
        roughness: roughness value (used as folder name)
        heightmap: the generated heightmap array
        template_folder: path to the template folder to copy
        output_base_dir: base directory where new folders are created
        
    Returns:
        path to the created folder
    """
    # Create folder name from roughness value
    folder_name = f"{roughness:.1f}"
    target_folder = os.path.join(output_base_dir, folder_name)
    
    # Remove existing folder if it exists
    if os.path.exists(target_folder):
        print(f"  Removing existing folder: {target_folder}")
        shutil.rmtree(target_folder)
    
    # Copy template folder
    if os.path.exists(template_folder):
        print(f"  Copying template folder to: {target_folder}")
        shutil.copytree(template_folder, target_folder)
    else:
        print(f"  WARNING: Template folder not found: {template_folder}")
        print(f"  Creating empty folder: {target_folder}")
        os.makedirs(target_folder, exist_ok=True)
    
    # Save heightmap as 1.png in the target folder
    png_path = os.path.join(target_folder, "1.png")
    Image.fromarray(heightmap, mode='L').save(png_path)
    print(f"  Saved heightmap to: {png_path}")
    
    return target_folder


# ================================================================
# === Visualization ===
# ================================================================

def create_comparison_plot(terrains, roughness_values, filename):
    """Create a side-by-side comparison plot of all terrain levels."""
    n_terrains = len(terrains)
    fig, axes = plt.subplots(1, n_terrains, figsize=(3 * n_terrains, 3))
    
    if n_terrains == 1:
        axes = [axes]
    
    for i, (roughness, heightmap) in enumerate(zip(roughness_values, terrains)):
        axes[i].imshow(heightmap, cmap='terrain', vmin=0, vmax=255)
        params = get_feature_params(roughness)
        axes[i].set_title(
            f'R={roughness:.1f}\n'
            f'pits={params["num_pits"]}, gaps={params["num_gaps"]}',
            fontsize=9
        )
        axes[i].axis('off')
    
    plt.suptitle('Unified Roughness Scaling: Noise + Pits/Gaps', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to: {filename}")


def create_3d_visualization(heightmap, roughness, filename):
    """Create a 3D surface plot of the terrain."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for faster plotting
    step = 4
    X_plot = np.arange(0, SIZE, step)
    Y_plot = np.arange(0, SIZE, step)
    X_mesh, Y_mesh = np.meshgrid(X_plot, Y_plot)
    Z_plot = heightmap[::step, ::step]
    
    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z_plot,
        cmap='terrain',
        linewidth=0,
        antialiased=True
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title(f'3D Terrain View (Roughness = {roughness:.1f})')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ================================================================
# === Main Execution ===
# ================================================================

def main():
    """Main function to generate all terrain levels."""
    
    print("=" * 60)
    print("Unified Terrain Generator")
    print("=" * 60)
    
    # Define roughness levels
    roughness_levels = [0.0, 0.2, 0.4, 0.7, 1.0, 1.4, 2.0]
    
    # Print scaling parameters for each level
    print("\n--- Feature Scaling Summary ---")
    for r in roughness_levels:
        print_feature_params(r)
    
    # Generate terrains
    print("\n--- Generating Terrains ---")
    terrains = []
    
    for roughness in roughness_levels:
        print(f"\nProcessing roughness = {roughness:.1f}")
        
        # Generate heightmap
        heightmap = generate_terrain(roughness, random_seed=42)
        terrains.append(heightmap)
        
        # Setup folder and save PNG
        setup_terrain_folder(
            roughness,
            heightmap,
            TEMPLATE_FOLDER,
            OUTPUT_BASE_DIR
        )
    
    # Create comparison plot
    print("\n--- Creating Visualizations ---")
    comparison_path = os.path.join(OUTPUT_BASE_DIR, "terrain_comparison_unified.png")
    create_comparison_plot(terrains, roughness_levels, comparison_path)
    
    # Create 3D visualization for a few key levels
    for roughness, heightmap in zip([0.0, 1.0, 2.0], [terrains[0], terrains[4], terrains[6]]):
        viz_path = os.path.join(OUTPUT_BASE_DIR, f"terrain_3d_roughness_{roughness:.1f}.png")
        create_3d_visualization(heightmap, roughness, viz_path)
        print(f"3D visualization saved to: {viz_path}")
    
    print("\n" + "=" * 60)
    print("All terrain files generated successfully!")
    print("=" * 60)
    
    # Print summary
    print("\nGenerated folders:")
    for r in roughness_levels:
        folder_path = os.path.join(OUTPUT_BASE_DIR, f"{r:.1f}")
        print(f"  {folder_path}/1.png")


if __name__ == "__main__":
    main()