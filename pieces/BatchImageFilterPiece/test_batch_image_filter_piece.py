from domino.testing import piece_dry_run
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import tempfile
import shutil


def test_batch_image_filter_piece():
    """Test the BatchImageFilterPiece with multiple images."""
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test images
        image_paths = []
        for i in range(3):
            # Create a simple test image (100x100 RGB)
            img_array = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = temp_path / f"test_image_{i}.png"
            img.save(img_path)
            image_paths.append(str(img_path))
        
        # Create test CSV
        csv_data = pd.DataFrame({
            'image_path': image_paths,
            'output_name': [f'output_{i}.png' for i in range(3)]
        })
        csv_path = temp_path / "test_images.csv"
        csv_data.to_csv(csv_path, index=False)
        
        # Run the piece
        input_data = dict(
            csv_file_path=str(csv_path),
            output_directory_name="test_filtered_images",
            sepia=True,
            contrast=True,
            output_type="file"
        )
        
        piece_output = piece_dry_run(
            piece_name="BatchImageFilterPiece",
            input_data=input_data
        )
        
        # Assertions
        assert piece_output is not None
        assert piece_output.get('processed_count') == 3
        assert piece_output.get('failed_count') == 0
        assert piece_output.get('results_csv_path').endswith('.csv')
        assert 'test_filtered_images' in piece_output.get('output_directory_path')


def test_batch_image_filter_piece_with_base64():
    """Test the BatchImageFilterPiece with base64 output."""
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a single test image
        img_array = np.ones((50, 50, 3), dtype=np.uint8) * 128  # Gray image
        img = Image.fromarray(img_array)
        img_path = temp_path / "test_image.png"
        img.save(img_path)
        
        # Create test CSV
        csv_data = pd.DataFrame({
            'image_path': [str(img_path)]
        })
        csv_path = temp_path / "test_single_image.csv"
        csv_data.to_csv(csv_path, index=False)
        
        # Run the piece with base64 output
        input_data = dict(
            csv_file_path=str(csv_path),
            output_directory_name="test_base64_output",
            black_and_white=True,
            output_type="both"
        )
        
        piece_output = piece_dry_run(
            piece_name="BatchImageFilterPiece",
            input_data=input_data
        )
        
        # Assertions
        assert piece_output is not None
        assert piece_output.get('processed_count') == 1
        assert piece_output.get('failed_count') == 0
        assert piece_output.get('base64_images_json_path') != '' 