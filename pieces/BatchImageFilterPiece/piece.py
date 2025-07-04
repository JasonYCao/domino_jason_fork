"""
BatchImageFilterPiece - Apply filters to multiple images from a CSV file.

This piece processes multiple images listed in a CSV file and applies the same set of filters to all of them.

CSV Format:
- Required column: 'image_path' - The path to each image file (can be absolute or relative)
- Optional column: 'output_name' - Custom name for the output file. If not provided, 
                                  the output name will be generated as '{original_name}_filtered.{extension}'

Example CSV:
    image_path,output_name
    /path/to/image1.jpg,my_custom_output1.jpg
    /path/to/image2.png,
    ./relative/path/image3.jpeg,filtered_result.jpeg

The piece will:
1. Read all image paths from the CSV
2. Apply the selected filters to each image
3. Save filtered images to an output directory
4. Generate a results CSV with processing status for each image
5. Optionally save base64 encoded images to a JSON file
"""

from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import pandas as pd
import os
import json


filter_masks = {
    'sepia': ((0.393, 0.769, 0.189), (0.349, 0.686, 0.168), (0.272, 0.534, 0.131)),
    'black_and_white': ((0.333, 0.333, 0.333), (0.333, 0.333, 0.333), (0.333, 0.333, 0.333)),
    'brightness': ((1.4, 0, 0), (0, 1.4, 0), (0, 0, 1.4)),
    'darkness': ((0.6, 0, 0), (0, 0.6, 0), (0, 0, 0.6)),
    'contrast': ((1.2, 0.6, 0.6), (0.6, 1.2, 0.6), (0.6, 0.6, 1.2)),
    'red': ((1.6, 0, 0), (0, 1, 0), (0, 0, 1)),
    'green': ((1, 0, 0), (0, 1.6, 0), (0, 0, 1)),
    'blue': ((1, 0, 0), (0, 1, 0), (0, 0, 1.6)),
    'cool': ((0.9, 0, 0), (0, 1.1, 0), (0, 0, 1.3)),
    'warm': ((1.2, 0, 0), (0, 0.9, 0), (0, 0, 0.8)),
}


class BatchImageFilterPiece(BasePiece):

    def apply_filters_to_image(self, image: Image.Image, filters: list) -> Image.Image:
        """Apply the selected filters to an image."""
        # Convert Image to NumPy array
        np_image = np.array(image, dtype=float)

        # Apply filters
        for filter_name in filters:
            np_mask = np.array(filter_masks[filter_name], dtype=float)
            for y in range(np_image.shape[0]):
                for x in range(np_image.shape[1]):
                    if np_image.shape[2] >= 3:  # Ensure we have at least RGB channels
                        rgb = np_image[y, x, :3]
                        new_rgb = np.dot(np_mask, rgb)
                        np_image[y, x, :3] = new_rgb
            # Clip values to be in valid range
            np_image = np.clip(np_image, 0, 255)

        # Convert back to uint8 and PIL image
        np_image = np_image.astype(np.uint8)
        return Image.fromarray(np_image)

    def process_single_image(self, image_path: str, output_name: str, output_dir: Path, 
                           filters: list, output_type: str) -> dict:
        """Process a single image and return the result."""
        result = {
            'input_path': image_path,
            'output_path': '',
            'base64_string': '',
            'status': 'success',
            'error': ''
        }

        try:
            # Try to open image from file path or base64 encoded string
            max_path_size = int(os.pathconf('/', 'PC_PATH_MAX'))
            if len(image_path) < max_path_size and Path(image_path).exists() and Path(image_path).is_file():
                image = Image.open(image_path)
            else:
                self.logger.info(f"Input image {image_path} is not a file path, trying to decode as base64 string")
                try:
                    decoded_data = base64.b64decode(image_path)
                    image_stream = BytesIO(decoded_data)
                    image = Image.open(image_stream)
                    image.verify()
                    image = Image.open(image_stream)
                except Exception:
                    raise ValueError(f"Input image {image_path} is not a file path or a base64 encoded string")

            # Apply filters
            filtered_image = self.apply_filters_to_image(image, filters)

            # Save to file
            if output_type == "file" or output_type == "both":
                output_path = output_dir / output_name
                filtered_image.save(output_path)
                result['output_path'] = str(output_path)

            # Convert to base64 string
            if output_type == "base64_string" or output_type == "both":
                buffered = BytesIO()
                filtered_image.save(buffered, format="PNG")
                result['base64_string'] = base64.b64encode(buffered.getvalue()).decode('utf-8')

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            self.logger.error(f"Error processing image {image_path}: {e}")

        return result

    def piece_function(self, input_data: InputModel):
        # Read CSV file
        try:
            df = pd.read_csv(input_data.csv_file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        # Validate CSV structure
        if 'image_path' not in df.columns:
            raise ValueError("CSV file must have an 'image_path' column")

        # Prepare filters list
        all_filters = []
        if input_data.sepia:
            all_filters.append('sepia')
        if input_data.black_and_white:
            all_filters.append('black_and_white')
        if input_data.brightness:
            all_filters.append('brightness')
        if input_data.darkness:
            all_filters.append('darkness')
        if input_data.contrast:
            all_filters.append('contrast')
        if input_data.red:
            all_filters.append('red')
        if input_data.green:
            all_filters.append('green')
        if input_data.blue:
            all_filters.append('blue')
        if input_data.cool:
            all_filters.append('cool')
        if input_data.warm:
            all_filters.append('warm')

        self.logger.info(f"Applying filters: {', '.join(all_filters) if all_filters else 'None'}")

        # Create output directory
        output_dir = Path(self.results_path) / input_data.output_directory_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each image
        results = []
        base64_images = {}
        processed_count = 0
        failed_count = 0

        for idx, row in df.iterrows():
            image_path = row['image_path']
            
            # Determine output name
            if 'output_name' in df.columns and pd.notna(row['output_name']):
                output_name = row['output_name']
            else:
                # Generate output name based on input file name
                input_path = Path(image_path)
                output_name = f"{input_path.stem}_filtered{input_path.suffix}"

            self.logger.info(f"Processing image {idx + 1}/{len(df)}: {image_path}")
            
            result = self.process_single_image(
                image_path, output_name, output_dir, 
                all_filters, input_data.output_type
            )
            
            results.append(result)
            
            if result['status'] == 'success':
                processed_count += 1
                if result['base64_string']:
                    base64_images[output_name] = result['base64_string']
            else:
                failed_count += 1

        # Create results CSV
        results_df = pd.DataFrame(results)
        results_csv_path = Path(self.results_path) / f"{input_data.output_directory_name}_results.csv"
        results_df.to_csv(results_csv_path, index=False)

        # Save base64 images to JSON if needed
        base64_json_path = ""
        if base64_images:
            base64_json_path = Path(self.results_path) / f"{input_data.output_directory_name}_base64.json"
            with open(base64_json_path, 'w') as f:
                json.dump(base64_images, f, indent=2)
            base64_json_path = str(base64_json_path)

        self.logger.info(f"Processing complete: {processed_count} successful, {failed_count} failed")

        # Display result (show first successful image if any)
        if processed_count > 0 and (input_data.output_type == "base64_string" or input_data.output_type == "both"):
            first_success = next((r for r in results if r['status'] == 'success' and r['base64_string']), None)
            if first_success:
                self.display_result = {
                    "file_type": "png",
                    "base64_content": first_success['base64_string'],
                    "file_path": first_success['output_path']
                }

        # Return output
        return OutputModel(
            output_directory_path=str(output_dir),
            results_csv_path=str(results_csv_path),
            processed_count=processed_count,
            failed_count=failed_count,
            base64_images_json_path=base64_json_path
        ) 