"""
Metadata handling for camera captures and video recordings
Handles EXIF embedding for images and metadata embedding for videos
"""

import json
import os
import subprocess
from datetime import datetime
from PIL import Image
import piexif


class MetadataHandler:
    """Handles metadata embedding for images and videos"""

    @staticmethod
    def get_camera_controls_metadata(source_type, source_info, settings, controls_cache):
        """
        Get current camera controls as metadata dictionary

        Args:
            source_type: Type of source ('camera', 'video', 'test')
            source_info: Dict with 'video_file' or 'camera_index'
            settings: Dict with 'resolution', 'zoom', 'rotation', 'perspective_correction'
            controls_cache: Cached camera controls dict

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        # Add timestamp
        metadata['capture_timestamp'] = datetime.now().isoformat()

        # Add source information
        metadata['source_type'] = source_type
        if source_type == 'video':
            metadata['video_file'] = source_info.get('video_file', 'unknown')
        elif source_type == 'camera':
            metadata['camera_index'] = source_info.get('camera_index', 0)

        # Add camera settings
        resolution = settings.get('resolution', (0, 0))
        metadata['resolution'] = f"{resolution[0]}x{resolution[1]}"
        metadata['zoom'] = settings.get('zoom', 1.0)
        metadata['rotation'] = settings.get('rotation', 0)

        # Add camera controls if available
        if controls_cache.get('available'):
            controls = controls_cache.get('controls', {})
            for name, info in controls.items():
                current_value = info.get('current')
                if current_value is not None:
                    metadata[f'camera_{name}'] = current_value

        # Add detection settings
        metadata['perspective_correction'] = settings.get('perspective_correction', False)

        return metadata

    @staticmethod
    def embed_image_metadata(filepath, metadata):
        """
        Embed metadata into image file as EXIF tags

        Args:
            filepath: Path to the image file
            metadata: Dictionary of metadata to embed
        """
        # Load image with PIL to add EXIF data
        img = Image.open(filepath)

        # Create EXIF data
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # Add metadata to UserComment (can store JSON)
        metadata_json = json.dumps(metadata)
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = metadata_json.encode('utf-8')

        # Add timestamp to DateTimeOriginal
        dt = datetime.now()
        datetime_str = dt.strftime("%Y:%m:%d %H:%M:%S")
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = datetime_str.encode('utf-8')
        exif_dict["0th"][piexif.ImageIFD.DateTime] = datetime_str.encode('utf-8')

        # Add camera make/model from metadata
        exif_dict["0th"][piexif.ImageIFD.Make] = b"Raspberry Pi"
        source_type = metadata.get('source_type', 'unknown')
        exif_dict["0th"][piexif.ImageIFD.Model] = f"Target Camera (Source: {source_type})".encode('utf-8')

        # Add image description with key settings
        description = f"Resolution: {metadata.get('resolution')}, Zoom: {metadata.get('zoom')}, Rotation: {metadata.get('rotation')}"
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')

        # Encode and save
        exif_bytes = piexif.dump(exif_dict)
        img.save(filepath, "jpeg", exif=exif_bytes, quality=95)

    @staticmethod
    def embed_video_metadata(video_filepath, metadata_filepath=None):
        """
        Embed metadata into video file using FFmpeg

        Args:
            video_filepath: Path to the video file
            metadata_filepath: Optional path to metadata JSON file (defaults to video_filepath.json)
        """
        try:
            # Read the metadata JSON file
            base_name = os.path.splitext(video_filepath)[0]
            if metadata_filepath is None:
                metadata_filepath = f"{base_name}.json"

            if not os.path.exists(metadata_filepath):
                print(f"Warning: Metadata file not found: {metadata_filepath}")
                return

            with open(metadata_filepath, 'r') as f:
                metadata = json.load(f)

            # Create temporary output file
            temp_filepath = f"{base_name}_temp{os.path.splitext(video_filepath)[1]}"

            # Build FFmpeg command with metadata tags
            cmd = ['ffmpeg', '-i', video_filepath, '-codec', 'copy']

            # Add standard metadata tags
            cmd.extend(['-metadata', 'title=Raspberry Pi Target Camera Recording'])

            # Add all metadata values as individual tags
            # FFmpeg metadata keys should not have spaces or special characters
            for key, value in metadata.items():
                # Convert the key to a valid metadata tag name
                # Replace underscores and make it more readable
                if key == 'capture_timestamp':
                    cmd.extend(['-metadata', f'date={value}'])
                    cmd.extend(['-metadata', f'{key}={value}'])
                elif key.startswith('camera_'):
                    # Add both with and without 'camera_' prefix
                    clean_key = key.replace('camera_', '')
                    cmd.extend(['-metadata', f'{clean_key}={value}'])
                    cmd.extend(['-metadata', f'{key}={value}'])
                elif key == 'video_file':
                    cmd.extend(['-metadata', f'source_file={value}'])
                    cmd.extend(['-metadata', f'{key}={value}'])
                elif key == 'camera_index':
                    cmd.extend(['-metadata', f'camera={value}'])
                    cmd.extend(['-metadata', f'{key}={value}'])
                else:
                    # Add all other metadata as-is
                    cmd.extend(['-metadata', f'{key}={value}'])

            # Add complete metadata as comment field (JSON string) for programmatic access
            metadata_json = json.dumps(metadata).replace('"', '\\"')  # Escape quotes
            cmd.extend(['-metadata', f'comment={metadata_json}'])

            # Output to temp file
            cmd.extend(['-y', temp_filepath])  # -y to overwrite without asking

            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Replace original file with metadata-embedded version
                os.replace(temp_filepath, video_filepath)
                print(f"Successfully embedded metadata into {os.path.basename(video_filepath)}")

                # Delete the JSON sidecar file since metadata is now embedded
                if os.path.exists(metadata_filepath):
                    os.remove(metadata_filepath)
                    print(f"Removed JSON sidecar file: {os.path.basename(metadata_filepath)}")
            else:
                print(f"Warning: FFmpeg failed to embed metadata: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)

        except Exception as e:
            print(f"Error embedding video metadata: {e}")
