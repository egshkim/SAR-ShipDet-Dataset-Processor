from UnifiedDatasetProcessor import *

# config = ProcessingConfig(
    
#     input_dir="/mnt/data/egshkim/ai/Dataset/DOTA/Dataset_for_VDIS/SAR/HRSID/HRSID_PNG",  # Directory containing both images and labels
#     output_dir="/mnt/data/egshkim/ai/Dataset/DOTA/Dataset_for_VDIS/SAR/KCGSA_Integrated_SARshipdet_Dataset/scripts_for_github/test/HRSID",
#     target_size=(1024, 1024),
#     coord_format=CoordinateFormat.RELATIVE,
#     visualize=True
# )
# processor = DatasetProcessor(config)
# processor.process_hrsid()

# config = ProcessingConfig(
    
#     input_dir="/mnt/data/egshkim/ai/Dataset/DOTA/Dataset_for_VDIS/SAR/Official-SSDD-OPEN/RBox_SSDD/voc_style",  # Directory containing both images and labels
#     output_dir="/mnt/data/egshkim/ai/Dataset/DOTA/Dataset_for_VDIS/SAR/KCGSA_Integrated_SARshipdet_Dataset/scripts_for_github/test/SSDD",
#     target_size=(1024, 1024),
#     coord_format=CoordinateFormat.RELATIVE,
#     visualize=True
# )
# processor = DatasetProcessor(config)
# processor.process_ssdd()

# config = ProcessingConfig(
    
#     input_dir="/mnt/data/egshkim/ai/Dataset/DOTA/Dataset_for_VDIS/SAR/SAR-Ship-Dataset/raw",  # Directory containing both images and labels
#     output_dir="/mnt/data/egshkim/ai/Dataset/DOTA/Dataset_for_VDIS/SAR/KCGSA_Integrated_SARshipdet_Dataset/scripts_for_github/test/SAR-Ship-Dataset",
#     target_size=(1024, 1024),
#     coord_format=CoordinateFormat.RELATIVE,
#     visualize=True
# )
# processor = DatasetProcessor(config)
# processor.process_sar_ship()

config = ProcessingConfig(
    
    input_dir="/mnt/data/egshkim/ai/Dataset/DOTA/Dataset_for_VDIS/SAR/SRSDD-V1.0",  # Directory containing both images and labels
    output_dir="/mnt/data/egshkim/ai/Dataset/DOTA/Dataset_for_VDIS/SAR/KCGSA_Integrated_SARshipdet_Dataset/scripts_for_github/test/SRSDD",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True
)
processor = DatasetProcessor(config)
processor.process_srsdd()
