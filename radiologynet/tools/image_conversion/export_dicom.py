import numpy as np
from pydicom import dcmread
from PIL import Image 
import os
import PIL.ImageOps

from radiologynet.tools.image_conversion.drop_policies import shape_policy, value_policy   

class ExportDcm():
    """
    Class that handles dicom export to images.
    """

    def __init__(self, dcm_path: str = None, export_dir: str = "./"):
        """
        Function that takes input path to dcm file and output path to folder where images will be saved.

        Args:
            * dcm_path, str --> path to dcm file that will be exported, Default None.
            * export_dir, str --> path to directory where the images will exported. Defould "./"
        """

        assert dcm_path != None, "dcm_path must be a string representing path to dcm file)"

        if export_dir == "./":
            self.export_dir = os.getcwd()
        else:
            self.export_dir = os.path.normpath(export_dir)

        # Load dcm data
        try:
            self.dcm_data = dcmread(dcm_path)
        except:
            print(f"Could not read dcm: {dcm_path}. Please check path!")
            return -1

        # Check for modality
        try:
            self.dcm_modality = self.dcm_data.Modality
        except:
            print(f"Could not get modality for dcm: {dcm_path}. Export unavailable")
            return -1

        # Check for pixeldata
        try:
            self.dcm_pixel_data = self.dcm_data.pixel_array
        except:
            print(f"Missing pixel data for dcm: {dcm_path}. Export unavailable")
            return -1

        # Get file name
        self.dcm_name = os.path.basename(dcm_path)
        
    def __apply_slope(self, image: np.array, intercept: int, slope: int) -> np.array:
        """

        Args:
            * image, np.array --> Input pixel data
            * intercept, int --> Intercept of the image
            * slope, int --> Slope of the image
        """            
        _image = image * slope + intercept
        return _image

    
    def __apply_window(self, image: np.array, bits: float, window_center: int, window_width: float) -> np.array:
        """
        Function that is applaying window center and window width to the pixel array.
        If window center and width are lists, only first value is taken. 

        Args:
            * image, int --> pixel array obtained by get_pixel_data
            * bits, int --> number of bits to export in ".png". Default 8
            * window_center, int --> window level/center
            * window_width, int --> window width
        """
        #_image = np.array(image)
        #print(np.max(_image))
        _max_pixel_intensity = pow(2, bits) - 1
        # Apply slope and intercept if avalable
        _out = np.piecewise(image, [image <= (window_center - (window_width)/2), image > (window_center + (window_width)/2)], 
                [0, _max_pixel_intensity, lambda image: (image - window_center + window_width/2)/window_width *_max_pixel_intensity])

        return _out

    
    def __export_MR(self, export_dir: str):
        """
        Function for MR export

        Args:
            * export_dir, str --> path to directory where images will be saved.
        
        """

        # Check for number of images
        _shape = self.dcm_pixel_data.shape
        if len(_shape) == 2:
            try:
                _windows_width = float(self.dcm_data.WindowWidth)
                _windows_center = float(self.dcm_data.WindowCenter)
            except:
                print("Missing windows width and windows center parameters. Skipping image")
                return -1

            try:
                _slope = float(self.dcm_data.RescaleSlope)
                _intercept = float(self.dcm_data.RescaleIntercept)
            except:
                _slope = 1
                _intercept = 0

            # Apply window and slope
            _image = self.dcm_pixel_data
            _image = self.__apply_slope(_image, _intercept, _slope)
            _image = self.__apply_window(_image, 8, _windows_center, _windows_width)

            # Export
            if shape_policy(pixel_data = _image, threshold = 0.1) \
                 and value_policy(pixel_data = _image,  threshold = 0.1):
                _image = Image.fromarray(np.uint8(_image))
                _image.save(os.path.join(export_dir, str(0) + ".png"))
            else:
                print(f"Value and policy check did not pass for: {self.dcm_name}. Skipping export for an image")
        else:
            for _index, _image in enumerate(self.dcm_pixel_data):
                # Obtain windows width, windows center, slope and intercept
                try:
                    _windows_width = float(self.dcm_data[0x5200, 0x9230][_index][0x28,0x9132][0]['WindowWidth'].value)
                    _windows_center = float(self.dcm_data[0x5200, 0x9230][_index][0x28,0x9132][0]['WindowCenter'].value)
                except:
                    print("Missing windows width and windows center parameters. Skipping image")
                    continue

                try:
                    _slope = float(self.dcm_data[0x5200, 0x9230][_index][0x28,0x9145][0]['RescaleSlope'].value)
                    _intercept = float(self.dcm_data[0x5200, 0x9230][_index][0x28,0x9145][0]['RescaleIntercept'].value)
                except:
                    _slope = 0
                    _intercept = 1
                
                # Apply window and slope
                _image = np.array(_image)
                _image = self.__apply_slope(_image, _intercept, _slope)
                _image = self.__apply_window(_image, 8, _windows_center, _windows_width)

                #Export
                if shape_policy(pixel_data = _image, threshold = 0.1) \
                    and value_policy(pixel_data = _image,  threshold = 0.1):
                    _image = Image.fromarray(np.uint8(_image))
                    _image.save(os.path.join(export_dir, str(_index) + ".png"))
                else:
                    print(f"Value and policy check did not pass for: {self.dcm_name}. Skipping export for an image")

    
    def __export_CR(self, export_dir: str):
        # Check for number of images
        _shape = self.dcm_pixel_data.shape

        if len(_shape) == 2:
            # Handle one image
            # Obtain windows width, windows center, slope and intercept
            try:
                _windows_width = float(self.dcm_data.WindowWidth)
                _windows_center = float(self.dcm_data.WindowCenter)
            except:
                print("Missing windows width and windows center parameters. Skipping image")
                return -1

            try:
                _slope = float(self.dcm_data.RescaleSlope)
                _intercept = float(self.dcm_data.RescaleIntercept)
            except:
                _slope = 1
                _intercept = 0
            
            # Apply window and slope
            _image = self.dcm_pixel_data
            _image = self.__apply_slope(_image, _intercept, _slope)
            _image = self.__apply_window(_image, 8, _windows_center, _windows_width)

            # Check for LUT inversion
            try:
                _lut = self.dcm_data[0x2050,0x0020]
                if _lut.value == 'INVERSE':
                    _image = np.abs(_image - 255)

            except:
                _image = np.abs(_image - 255)

            #Export
            if shape_policy(pixel_data = _image, threshold = 0.1) \
                 and value_policy(pixel_data = _image,  threshold = 0.1):
                _image = Image.fromarray(np.uint8(_image))
                _image.save(os.path.join(export_dir, str(0) + ".png"))
            else:
                print(f"Value and policy check did not pass for: {self.dcm_name}. Skipping export for an image")

    def __export_NM(self, export_dir: str):
        """
        Function for NM export

        Args:
            * export_dir, str --> path to directory where images will be saved.
        
        """

        # Check for number of images
        _shape = self.dcm_pixel_data.shape

        if len(_shape) == 2:
            try:
                _windows_width = float(self.dcm_data.WindowWidth)
                _windows_center = float(self.dcm_data.WindowCenter)
            except:
                print("Missing windows width and windows center parameters. Skipping image")
                return -1

            try:
                _slope = float(self.dcm_data.RescaleSlope)
                _intercept = float(self.dcm_data.RescaleIntercept)
            except:
                _slope = 1
                _intercept = 0
            
            # Apply window and slope
            _image = self.dcm_pixel_data
            _image = self.__apply_slope(_image, _intercept, _slope)
            _image = self.__apply_window(_image, 8, _windows_center, _windows_width)

            # Export
            if shape_policy(pixel_data = _image, threshold = 0.1) \
                 and value_policy(pixel_data = _image,  threshold = 0.1):
                _image = Image.fromarray(np.uint8(_image))
                _image.save(os.path.join(export_dir, str(0) + ".png"))
            else:
                print(f"Value and policy check did not pass for: {self.dcm_name}. Skipping export for an image")
        else:
            for _index, _image in enumerate(self.dcm_pixel_data):
                # Obtain windows width, windows center, slope and intercept
                try:
                    _windows_width = float(self.dcm_data.WindowWidth)
                    _windows_center = float(self.dcm_data.WindowCenter)
                except:
                    print("Missing windows width and windows center parameters. Skipping image")
                    return -1

                try:
                    _slope = float(self.dcm_data.RescaleSlope)
                    _intercept = float(self.dcm_data.RescaleIntercept)
                except:
                    _slope = 1
                    _intercept = 0
                
                # Apply window and slope
                _image = np.array(_image)
                _image = self.__apply_slope(_image, _intercept, _slope)
                _image = self.__apply_window(_image, 8, _windows_center, _windows_width)

                #Export
                if shape_policy(pixel_data = _image, threshold = 0.1) \
                    and value_policy(pixel_data = _image,  threshold = 0.1):
                    _image = Image.fromarray(np.uint8(_image))
                    _image.save(os.path.join(export_dir, str(_index) + ".png"))
                else:
                    print(f"Value and policy check did not pass for: {self.dcm_name}. Skipping export for an image")


    
    def __export_CT(self, export_dir: str):
        # Check for number of images
        _shape = self.dcm_pixel_data.shape

        if len(_shape) == 2:
            # Handle one image
            # Obtain windows width, windows center, slope and intercept
            _single_window = True
            try:
                _windows_width_list = []
                _windows_center_list = []
                _windows_width_list.append(float(self.dcm_data.WindowWidth))
                _windows_center_list.append(float(self.dcm_data.WindowCenter))
            except:
                _single_window = False

            if _single_window == False:
                try:
                    _windows_width_list = self.dcm_data.WindowWidth
                    _windows_center_list = self.dcm_data.WindowCenter
                    _exp = self.dcm_data[0x28, 0x1055] #Safety
                except:
                    print("Window center and window width are invalid. Skipping image")
                    return -1
            try:
                _slope = float(self.dcm_data.RescaleSlope)
                _intercept = float(self.dcm_data.RescaleIntercept)
            except:
                _slope = 1
                _intercept = 0
            
            _index = 0

            for _windows_center, _windows_width in zip(_windows_center_list, _windows_width_list):
                # Apply window and slope
                _image = self.dcm_pixel_data
                _image = self.__apply_slope(_image, _intercept, _slope)
                _image = self.__apply_window(_image, 8, float(_windows_center), float(_windows_width))

                #Export
                if shape_policy(pixel_data = _image, threshold = 0.1) \
                    and value_policy(pixel_data = _image,  threshold = 0.1):
                    _image = Image.fromarray(np.uint8(_image))
                    _image.save(os.path.join(export_dir, str(_index) + ".png"))
                    _index += 1
                else:
                    print(f"Value and policy check did not pass for: {self.dcm_name}. Skipping export for an image")


    def __export_RF(self, export_dir: str):
        # Check for number of images
        _shape = self.dcm_pixel_data.shape

        if len(_shape) == 2:
            # Handle one image
            # Obtain windows width, windows center, slope and intercept
            try:
                _windows_width = float(self.dcm_data.WindowWidth)
                _windows_center = float(self.dcm_data.WindowCenter)
            except:
                print("Missing windows width and windows center parameters. Skipping image")
                return -1

            try:
                _slope = float(self.dcm_data.RescaleSlope)
                _intercept = float(self.dcm_data.RescaleIntercept)
            except:
                _slope = 1
                _intercept = 0
            
            # Apply window and slope
            _image = self.dcm_pixel_data
            _image = self.__apply_slope(_image, _intercept, _slope)
            _image = self.__apply_window(_image, 8, _windows_center, _windows_width)

            #Export
            if shape_policy(pixel_data = _image, threshold = 0.1) \
                    and value_policy(pixel_data = _image,  threshold = 0.1):
                _image = Image.fromarray(np.uint8(_image))
                _image.save(os.path.join(export_dir, str(0) + ".png"))
            else:
                print(f"Value and policy check did not pass for: {self.dcm_name}. Skipping export for an image")

    def __export_XA(self, export_dir: str):
        # Check for number of images
        _shape = self.dcm_pixel_data.shape
        # Obtain windows width, windows center, slope and intercept
        _single_window = True
        try:
            _windows_width_list = []
            _windows_center_list = []
            _windows_width_list.append(float(self.dcm_data.WindowWidth))
            _windows_center_list.append(float(self.dcm_data.WindowCenter))
        except:
            _single_window = False

        if _single_window == False:
            try:
                _windows_width_list = self.dcm_data.WindowWidth
                _windows_center_list = self.dcm_data.WindowCenter
            except:
                print("Window center and window width are invalid. Skipping image")
                return -1

        try:
            _slope = float(self.dcm_data.RescaleSlope)
            _intercept = float(self.dcm_data.RescaleIntercept)
        except:
            _slope = 1
            _intercept = 0
        if len(_shape) == 2:
            # Handle one image            
            # Apply window and slope
            _index = 0
            for _windows_center, _windows_width in zip(_windows_center_list, _windows_width_list):
                _image = self.dcm_pixel_data
                _image = self.__apply_slope(_image, _intercept, _slope)
                _image = self.__apply_window(_image, 8, _windows_center, _windows_width)

                #Export
                if shape_policy(pixel_data = _image, threshold = 0.1) \
                        and value_policy(pixel_data = _image,  threshold = 0.1):
                    _image = Image.fromarray(np.uint8(_image))
                    _image.save(os.path.join(export_dir, str(_index) + ".png"))
                    _index += 1
                else:
                    print(f"Value and policy check did not pass for: {self.dcm_name}. Skipping export for an image")
 
        else:
            # For each img and each window
            _index = 0
            for _image in self.dcm_pixel_data:
                for _windows_center, _windows_width in zip(_windows_center_list, _windows_width_list):
                    # Apply window and slope
                    _image = np.array(_image)
                    _image = self.__apply_slope(_image, _intercept, _slope)
                    _image = self.__apply_window(_image, 8, _windows_center, _windows_width)

                    #Export
                    if shape_policy(pixel_data = _image, threshold = 0.1) \
                        and value_policy(pixel_data = _image,  threshold = 0.2):
                        _image = Image.fromarray(np.uint8(_image))
                        _image.save(os.path.join(export_dir, str(_index) + ".png"))
                        _index += 1
                    else:
                        print(f"Value and policy check did not pass for: {self.dcm_name}. Skipping export for an image")


    
    def export(self):
        """
        Function that exports images to file.
        """

        # Create export folders
        # Check if output dir exists
        if os.path.exists(self.export_dir) == False:
            print(self.export_dir)
            os.makedirs(self.export_dir)
        
        # Create study folder
        _study_dir = os.path.join(self.export_dir, f'conversion_{self.dcm_modality}', self.dcm_name[:-7], (self.dcm_modality + "_"+ self.dcm_name))
        print(_study_dir)
        
        if os.path.exists(_study_dir) == False:
            os.makedirs(_study_dir)
        
        if self.dcm_modality == 'MR':
            self.__export_MR(_study_dir)

        if self.dcm_modality == 'CR':
            self.__export_CR(_study_dir)

        if self.dcm_modality == 'NM':
            self.__export_NM(_study_dir)

        if self.dcm_modality == 'CT':
            self.__export_CT(_study_dir)

        if self.dcm_modality == 'RF':
            self.__export_RF(_study_dir)

        if self.dcm_modality == 'XA':
            self.__export_XA(_study_dir)
        
        # Delete dir if empty
        if os.path.isdir(_study_dir) and not os.listdir(_study_dir) :
            os.rmdir(_study_dir )
        
