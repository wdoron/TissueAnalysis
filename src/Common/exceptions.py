import enum

class ImageLoadError(Exception):
    """Custom exception raised when image loading fails."""
    pass

# class ImageHasMissingDataErrorType(enum.Enum):
#     DAB_IMAGE_FLAT = "Dab_Image_Flat"
#     ANALYZE_NO_CELLS = "Analyze_No_Cells"

# class ImageHasMissingDataError(Exception):
#     def __init__(self, error_type: ImageHasMissingDataErrorType, more_info: str = ""):
#         self.error_type = error_type
#         super().__init__(f"Image has missing data: type={error_type.value}, info={more_info}")
#         self.more_info = more_info
