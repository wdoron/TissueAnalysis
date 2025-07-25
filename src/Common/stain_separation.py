import numpy as np


class StainSeparation:
    def __init__(self):
        # Define the stain vectors for H-DAB
        # Values are taken from literature (e.g., Ruifrok and Johnston)

        self.hematoxylin = np.array([0.650, 0.704, 0.286])
        self.dab = np.array([0.268, 0.570, 0.776])
        # Compute the third (residual) stain vector as the cross product of the first two
        self.residual = np.cross(self.hematoxylin, self.dab)

        # Normalize the stain vectors to unit length
        self.hematoxylin = self.hematoxylin / np.linalg.norm(self.hematoxylin)
        self.dab = self.dab / np.linalg.norm(self.dab)
        self.residual = self.residual / np.linalg.norm(self.residual)

        # Build a 3x3 stain matrix with the stain vectors as columns:
        # Column 0: Hematoxylin, Column 1: DAB, Column 2: Residual.
        self.stain_matrix = np.column_stack((self.hematoxylin, self.dab, self.residual))

    def get_stain_channels(self, img):
        """
        Perform color deconvolution on an H-DAB stained RGB image.

        Parameters:
            img (numpy.ndarray): An RGB image with shape (H, W, 3) and pixel values in [0, 255].

        Returns:
            tuple: (hematoxylin_channel, dab_channel, residual_channel) as uint8 images.
        """
        # Convert image to float for computation
        img = img.astype(np.float64)

        # Replace zeros to avoid log(0)
        img[img == 0] = 1

        # Compute the optical density (OD) image.
        # Using the formula: OD = -log((I + 1) / 256)
        # (Adding 1 avoids log(0); 256 approximates the maximum intensity.)
        OD = -np.log((img + 1) / 256)

        # Reshape the OD image to (3, num_pixels)
        h, w, _ = OD.shape
        OD_reshaped = OD.reshape((-1, 3)).T  # shape: (3, h*w)

        # Compute the pseudo-inverse of the stain matrix
        pinv = np.linalg.pinv(self.stain_matrix)

        # Compute the concentrations for each stain.
        # Each column in 'concentrations' corresponds to the amount of a stain.
        concentrations = np.dot(pinv, OD_reshaped)  # shape: (3, h*w)

        # Reshape concentrations back to (3, h, w)
        concentrations = concentrations.reshape((3, h, w))

        # Reconstruct the stain images by inverting the OD transformation:
        # I_stain = 255 * exp(-concentration)
        hematoxylin_channel = 255 * np.exp(-concentrations[0])
        dab_channel = 255 * np.exp(-concentrations[1])
        residual_channel = 255 * np.exp(-concentrations[2])

        # Ensure the pixel values fall within [0, 255] and convert to uint8
        hematoxylin_channel = np.clip(hematoxylin_channel, 0, 255).astype(np.uint8)
        dab_channel = np.clip(dab_channel, 0, 255).astype(np.uint8)
        residual_channel = np.clip(residual_channel, 0, 255).astype(np.uint8)

        hematoxylin_channel = 255 - hematoxylin_channel
        dab_channel = 255 - dab_channel
        residual_channel = 255 - residual_channel

        return hematoxylin_channel, dab_channel, residual_channel
