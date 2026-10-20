import zipfile
import pydicom
import io
import pathlib
import numpy as np
from tqdm import tqdm
from pathlib import Path

# V1 Zip Unpacking Function
def PPMIunzip(zipfolderpath, additionalzips=None, returnscan=False):
    if additionalzips == None:
        full_data = [zipfolderpath]
    else:
        full_data = [zipfolderpath, additionalzips]

    # Check for existence of all files before unpacking
    for n, zipfolder in enumerate(full_data):
        zip_path = Path(zipfolder)
        zip_check = zip_path.exists()
        if zip_check == True:
            print(f"File {n+1}: {zipfolder} exists.")
            continue
        else:
            raise Exception(f"File {zipfolder} does not exist.")

    # Create empty array for data storage
    PPMI_complete = []

    # Iterate through given files
    for n, zipfolder in enumerate(full_data):
        # Iterating files and displaying name
        file_name = full_data[n]
        print(f"Checking file {n+1} for DICOM files...")
        
        # with statement allows file closing for resource management
        with zipfile.ZipFile(file_name, "r") as archive:
        
            # Find all DICOM files inside the ZIP
            dicom_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".dcm")
            ]
    
            # Setting number of files
            fileno = len(dicom_files)
            
            # Print number of found DICOM files
            print(f"Found {fileno} DICOM files.")
    
            # Setting Empty array to append with data
            #PPMI_array = np.zeros((len(dicom_files)))
            PPMI_array = np.empty((fileno,5), dtype=object)
            #PPMI_array = {}
            
            # Iterating on dataset
            with tqdm(total=fileno, desc=f"Processing file {n+1} scan data...") as pbar:
                for i, name in enumerate(dicom_files):
                    # Get the DICOM file contents from the zip folder
                    with archive.open(name, "r") as content:
                        ds = pydicom.dcmread(content)
                    #print(ds.shape)
    
                    # Finding patient ID number
                    pat_no = name.split('/')[1]
                    
                    # Return relevant data
                    try:
                        date = ds[0x0008,0x0020].value # Date of scan
                        year = str(date)[:4] # Year of scan
                        #print(year)
                        model = ds[0x0008, 0x1090].value # Scanner model
                        #print(f'Patient Number: {pat_no}, Scan Date: {date}, Scan Year: {year}, Scanner Model: {model}')
                    except:
                        pbar.update(1)
                        print(f"Skipping formatting exception: patient {pat_no}.")
                        continue

                    if returnscan == True:
                        # Access the pixel data
                        pixel_array = ds.pixel_array

                        # Append Scan data
                        PPMI_array[i,3] = pixel_array
                        PPMI_array[i,4] = ds                        
        
                    # Append to numpy data array
                    #PPMI_array[i,:] = [pat_no, year, model, pixel_array]
                    PPMI_array[i,0] = pat_no
                    PPMI_array[i,1] = year
                    PPMI_array[i,2] = model
    
                    # Update progress bar
                    pbar.update(1)
        PPMI_complete.append(PPMI_array)
        
    PPMI_complete = np.concatenate(PPMI_complete, axis=0)
    print("Processing complete.")
    return(PPMI_complete)
