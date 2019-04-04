"""
Created on Thu Apr 04 10:42:15 2019

@author: vane1601
"""

__author__ = """Elijah EW Van Houten"""
__email__ = 'eew.vanhouten@usherbrooke.ca'
__version__ = '0.1.0'

def load_mr_phase_select():
    root=Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ph_img = []
    outfile_file_names = []
    outfile_file_types = []
    phase_loop = True
    num_phase_data = 0
    while phase_loop:    
        root.ph_filename =  filedialog.askopenfilename(initialdir = "/",title = 'Select Phase Image File # '+str(num_phase_data+1),filetypes = (('NIfTI files','*.nii.gz'),('PAR files','*.PAR'),('all files','*.*')))
        if root.ph_filename:
            outfile_file_names.append(root.ph_filename)
            outfile_file_types.append('pha')
            if '.PAR' in root.ph_filename:
                print('Processing PAR/REC Files:')
                nifti=False
                parrec=True
                try:
                    ph_img.append(nib.load(root.ph_filename,strict_sort=True))
                except IOError:
                    print('Phase Data File Load Error!!! Invalid data file selected!!!')
            else:
                print('Processing NIfTI Files:')
                nifti=True
                try:
                    ph_img.append(nib.load(root.ph_filename))
                except:
                    print('Phase Data File Load Error!!! Invalid data file selected!!!')
            print_info('Phase Image File Name',root.ph_filename)
            num_phase_data += 1
        else:
            print('No Phase File Selected!!!')
            phase_loop = False
    root.destroy

def load_mr_mag_select():
    root=Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if nifti:
        root.mag_filename =  filedialog.askopenfilename(initialdir = "/",title = 'Select Magnitude Image File',filetypes = (('NIfTI files','*.nii.gz'),('all files','*.*')))
        if root.mag_filename:
            outfile_file_names.append(root.mag_filename)
            outfile_file_types.append('mag')
            print_info('Magnitude Image File Name',root.mag_filename)
            try:
                mag_img = nib.load(root.mag_filename)
            except:
                print('Phase Data File Load Error!!! Invalid data file selected!!!')
        else:
            print('No Magnitude File Selected!!!')          
    root.destroy
