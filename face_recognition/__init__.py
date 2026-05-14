__author__ = "Adam Geitgey"
__email__ = "ageitgey@gmail.com"
__version__ = "1.4.0"

from .api import (  # noqa: F401
    load_image_file,
    face_locations,
    batch_face_locations,
    face_landmarks,
    face_encodings,
    compare_faces,
    face_distance,
)
