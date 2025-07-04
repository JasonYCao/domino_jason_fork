from pydantic import BaseModel, Field
from enum import Enum


class OutputTypeType(str, Enum):
    """
    Output type for the result images
    """
    file = "file"
    base64_string = "base64_string"
    both = "both"


class InputModel(BaseModel):
    csv_file_path: str = Field(
        description='Path to CSV file containing image paths. The CSV should have at least an "image_path" column. Optionally, it can have an "output_name" column for custom output names.',
        json_schema_extra={
            "from_upstream": "always"
        }
    )
    output_directory_name: str = Field(
        default="filtered_images",
        description='Name of the output directory where filtered images will be saved.',
    )
    sepia: bool = Field(
        default=False,
        description='Apply sepia effect to all images.',
    )
    black_and_white: bool = Field(
        default=False,
        description='Apply black and white effect to all images.',
    )
    brightness: bool = Field(
        default=False,
        description='Apply brightness effect to all images.',
    )
    darkness: bool = Field(
        default=False,
        description='Apply darkness effect to all images.',
    )
    contrast: bool = Field(
        default=False,
        description='Apply contrast effect to all images.',
    )
    red: bool = Field(
        default=False,
        description='Apply red effect to all images.',
    )
    green: bool = Field(
        default=False,
        description='Apply green effect to all images.',
    )
    blue: bool = Field(
        default=False,
        description='Apply blue effect to all images.',
    )
    cool: bool = Field(
        default=False,
        description='Apply cool effect to all images.',
    )
    warm: bool = Field(
        default=False,
        description='Apply warm effect to all images.',
    )
    output_type: OutputTypeType = Field(
        default=OutputTypeType.file,
        description='Format of the output images. Options are: `file`, `base64_string`, `both`. Note: for batch processing, "file" is recommended.',
    )


class OutputModel(BaseModel):
    output_directory_path: str = Field(
        description='Path to the output directory containing all filtered images.',
    )
    results_csv_path: str = Field(
        description='Path to the CSV file containing the results summary with input paths, output paths, and processing status.',
    )
    processed_count: int = Field(
        description='Number of images successfully processed.',
    )
    failed_count: int = Field(
        description='Number of images that failed to process.',
    )
    base64_images_json_path: str = Field(
        default='',
        description='Path to JSON file containing base64 encoded images (only when output_type includes base64_string).',
    ) 