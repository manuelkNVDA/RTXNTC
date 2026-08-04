# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Requirements:
# -------------
# - This script need to be called within UE5. Please refer to UE5 Python plugin for details about running python scripts.
# - This script requires the NTC-CLI standalone tool. Make sure to set the 'path_ntc_cli' to the compiled binary path.

# Usage Instructions:
# -------------------
# - Modify values under the 'General Settings' to fit the local setup
# - Modify values under the 'NTC Compression Settings' to tweak compression quality/size
# - Under the 'UE5 Project Scripting' section...
# -- Create NTC packages by calling one of the following methods:
#       'ntc_pack_by_name' to generate NTC packages based on textures naming convention
#       'ntc_pack_by_folder' to generate NTC packages based on project folders structure
# -- Call the 'ue_export_compress_decompress_packs' method to have a set of textures compressed with NTC
# -- Call the 'ue_import_texture_packs' method to re-import NTC compressed textures in UE5.
# -- Call the 'ue_generate_stats' method to analyse texture packages and generate size reports for BC vs NTC

# Imports
import os
import platform
import math
import subprocess
import unreal
from struct import pack
from dataclasses import dataclass

# Globals
ue_asset_export_task = unreal.AssetExportTask()
ue_asset_import_task = unreal.AssetImportTask()
ue_asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
ue_syslib = unreal.SystemLibrary()
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# General Settings
ext_sdr_file = '.png'
ext_hdr_file = '.exr'
folder_ntc = 'ntc'
suffix_delimt = '_' # Can be modified depending on the naming convention used in the project
path_csv = os.path.join(root, 'texture_data.csv') # Modify to specify stat output

# NTC Compression Settings - passed directly to ntc-cli
ntc_bpp = 2.0
ntc_compression_args = f'--bitsPerPixel {ntc_bpp} --trainingSteps 80000'

# Find the NTC-CLI tool (architecture-aware; mirrors the CMake NTC_BINARY_DIR logic)
if os.name == 'nt':
    _bin_arch = 'bin/windows-arm64' if platform.machine().lower() in ('arm64', 'aarch64') else 'bin/windows-x64'
    path_ntc_cli = os.path.join(_bin_arch, 'ntc-cli.exe')
else:
    path_ntc_cli = 'bin/linux-x64/ntc-cli'
path_ntc_cli = os.path.join(root, path_ntc_cli)

# Data Classes
@dataclass
class NTCPack:
    name: str
    directory: str
    assets: list[str]

@dataclass
class NTCPackageStats:
    package_name: str
    package_size_bc: float
    package_size_ntc: float
    assets_names: list[str]
    assets_compressmode: list[str]
    assets_dims_x: list[int]
    assets_dims_y: list[int]
    assets_sizes: list[float]
    assets_count: int


# -----------------------------------
# Functions for processing file names
# -----------------------------------
    
# Get the base file name without the texture type seprated by the last the delimiter character
def filename_getbase(filename):
    i = filename.rfind(suffix_delimt)
    if(i != -1):
        return filename[:i]
    
# Get the file name with '.' delimiter as required by the standalone tool
def filename_getdotdelimt(filename):
    i = filename.rfind(suffix_delimt)
    if(i != -1):
        filename_chars = list(filename)
        filename_chars[i] = '.'
        return ''.join(filename_chars)

# ----------------------------------------------------
# Get assets of type Texture2D in a specified paths
# ----------------------------------------------------
def ue_get_assets(paths=[], assets_types=['Texture2D']):
    reg = unreal.AssetRegistryHelpers.get_asset_registry()
    filt = unreal.ARFilter(
        package_paths=paths,
        class_names=assets_types,
        recursive_paths=True,
        include_only_on_disk_assets=True
    )
    return reg.get_assets(filt)

# --------------------------------------------------------------
# Functions for analyzing texture files and generating NTC packs
# based on naming convention or folder structure
# --------------------------------------------------------------

# Generate ntc packages based on naming convention
def ntc_pack_by_name(paths=['/Game']):
    ntc_packs = []
    assets_names = []
    ue_assets = ue_get_assets(paths)
    for ue_asset in ue_assets:
        asset_fullname = str(ue_asset.asset_name)
        asset_basename = filename_getbase(asset_fullname)
        assets_names.append(asset_basename)
        
    unique_names = [*set(assets_names)]
    for unique_asset_name in unique_names:
        asset_indices = [i for i, x in enumerate(assets_names) if x == unique_asset_name]
        asset_package = [ue_assets[index] for index in asset_indices]
        asset_path = ue_syslib.get_system_path(asset_package[0].get_asset())
        asset_directory = os.path.dirname(asset_path)
        ntc_packs.append(NTCPack(unique_asset_name, os.path.join(asset_directory, unique_asset_name), asset_package))

    return ntc_packs

# Generate ntc packages based on folder structure
def ntc_pack_by_folder(paths=['/Game']):
    ntc_packs = []
    assets_folders = []    
    ue_assets = ue_get_assets(paths)
    for ue_asset in ue_assets:
        asset_path = ue_syslib.get_system_path(ue_asset.get_asset())
        asset_directory = os.path.dirname(asset_path)
        assets_folders.append(asset_directory)
    
    unique_folders = [*set(assets_folders)]
    for unique_asset_folder in unique_folders:
        asset_indices = [i for i, x in enumerate(assets_folders) if x == unique_asset_folder]
        asset_package = [ue_assets[index] for index in asset_indices]
        package_name = filename_getbase(str(asset_package[0].asset_name))        
        ntc_packs.append(NTCPack(package_name, unique_asset_folder, asset_package))
        
    return ntc_packs

# -------------------------------------------------
# Functions for importing raw images to UE textures
# -------------------------------------------------

# Get raw image format based on Texture2D compression settings
def ue_get_rawimage_ext(ue_asset):
    texture = ue_asset.get_asset()
    if (texture.compression_settings == unreal.TextureCompressionSettings.TC_HDR_COMPRESSED):
        return ext_hdr_file
    else:
        return ext_sdr_file

# Re-importing a texture from a raw image
def ue_reimport_texture(asset, path):
    # get color space of original texture
    texture=asset.get_asset()
    srgb=texture.srgb
    compression_settings=texture.compression_settings
    disk_path = path
    import_path = os.path.join(os.path.dirname(path), str(asset.asset_name)) + ue_get_rawimage_ext(asset)
    os.rename(disk_path, import_path)
    ue_asset_import_task.automated=True    
    ue_asset_import_task.destination_path=asset.package_path
    #ue_asset_import_task.destination_name=str(asset.asset_name) # not working, renaming raw file as a workaround
    ue_asset_import_task.filename=import_path   
    ue_asset_import_task.replace_existing=True
    ue_asset_import_task.save=False
    ue_asset_tools.import_asset_tasks([ue_asset_import_task])
    os.rename(import_path, disk_path)
    
    # apply color space after re-importing
    texture.srgb=srgb
    texture.compression_settings=compression_settings

# Re-import textures in one NTC Package
def ue_reimport_texture_package(ntc_pack, is_ntc = True):
    for ue_asset in ntc_pack.assets:
        asset_name = str(ue_asset.asset_name)
        fixed_name = filename_getdotdelimt(asset_name)
        if is_ntc:
            import_folder = os.path.join(ntc_pack.directory, folder_ntc)
        else:
            import_folder = ntc_pack.directory

        if fixed_name is not None:
            import_path = os.path.join(import_folder, fixed_name + ue_get_rawimage_ext(ue_asset))
            if(os.path.exists(import_path)):
                ue_reimport_texture(ue_asset, import_path)
            else:
                print(import_path + ' not found. Texture import skipped.')

# Re-import all textures in NTC packages. By default it import NTC compressed images
def ue_import_texture_packs(ntc_packs, import_ntc=True):
    ntc_packs_count = len(ntc_packs)
    # re-import NTC raw images
    with unreal.ScopedSlowTask(ntc_packs_count, 'Re-importing Textures') as slow_task:
        slow_task.make_dialog(True)
        for i in range(ntc_packs_count):
            if slow_task.should_cancel():
                return
            ue_reimport_texture_package(ntc_packs[i], import_ntc)
            ue_save_packages()
            slow_task.enter_progress_frame(1)

    print('Done importing textures')

# -----------------------
# NTC-CLI standalone tool
# -----------------------    
def ntc_recompress_pack(package_dir):
    ntc_folder = os.path.join(package_dir, folder_ntc)
    if not os.path.exists(ntc_folder):
        os.makedirs(ntc_folder)
    args = path_ntc_cli + ' --loadImages=' + package_dir + ' --saveImages=' + ntc_folder + ' --compress --decompress ' + ntc_compression_args
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    subprocess.call(args, startupinfo=si)

# -------------------------------------------------
# Functions for exporting UE textures to raw images
# -------------------------------------------------

# Exporting a Texture2D asset to a raw image (e.g. png, exr, ... etc)
def ue_export_texture(texture, path):
    ue_asset_export_task.filename=path
    ue_asset_export_task.object=texture
    unreal.Exporter.run_asset_export_task(ue_asset_export_task)

# Exporting textures in one NTC package
def ue_export_texture_package(ntc_pack):
    for ue_asset in ntc_pack.assets:
        fixed_name = filename_getdotdelimt(str(ue_asset.asset_name))
        file_ext = ue_get_rawimage_ext(ue_asset)
        if fixed_name is not None:
            export_path = os.path.join(ntc_pack.directory, fixed_name + file_ext)
            ue_export_texture(ue_asset.get_asset(), export_path)

# Exporting all textures in NTC packages
def ue_export_compress_decompress_packs(ntc_packs):
    # Get user confirmation to process assets
    ntc_packs_count = len(ntc_packs)    
    dialog = unreal.EditorDialog()
    user_confirmation = dialog.show_message('Neural Texture Pipeline', 'Found {} Textures.\nThis might take time to process. Continue?'.format(ntc_packs_count), unreal.AppMsgType.YES_NO, unreal.AppReturnType.NO)
    if(user_confirmation == unreal.AppReturnType.NO):
        return

    # Export texture assets to raw images
    with unreal.ScopedSlowTask(ntc_packs_count, 'Exporting Textures') as slow_task:
        slow_task.make_dialog(True)
        for i in range(ntc_packs_count):
            if slow_task.should_cancel():
                return
            ue_export_texture_package(ntc_packs[i])
            slow_task.enter_progress_frame(1)
    
    # Compress exported images to NTC
    with unreal.ScopedSlowTask(ntc_packs_count, 'Compressing/Decompressing Textures') as slow_task:
        slow_task.make_dialog(True)
        for i in range(ntc_packs_count):
            if slow_task.should_cancel():
                return
            ntc_recompress_pack(ntc_packs[i].directory)
            slow_task.enter_progress_frame(1)

# Saving Modified Assets
def ue_save_packages():
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(
        save_map_packages=False,
        save_content_packages=True
    )

# ------------------------------------------
# Functions for generating statistics report
# including sizes and compression ratios
# ------------------------------------------    

# Get number of channels for each texture type    
def ue_get_texture_channel_count(texture):
    channels_count = 4 # default channel count
    mode = texture.compression_settings
    if mode == unreal.TextureCompressionSettings.TC_NORMALMAP:
        channels_count = 2
    if mode == unreal.TextureCompressionSettings.TC_GRAYSCALE:
        channels_count = 1
    if mode == unreal.TextureCompressionSettings.TC_DISPLACEMENTMAP:
        channels_count = 1
    if mode == unreal.TextureCompressionSettings.TC_ALPHA:
        channels_count = 1
    if mode == unreal.TextureCompressionSettings.TC_DISTANCE_FIELD_FONT:
        channels_count = 1
    if mode == unreal.TextureCompressionSettings.TC_HDR_COMPRESSED:
        channels_count = 3
    if mode == unreal.TextureCompressionSettings.TC_HALF_FLOAT:
        channels_count = 1
    if mode == unreal.TextureCompressionSettings.TC_SINGLE_FLOAT:
        channels_count = 1

    if channels_count == 4:
        alpha_disabled = texture.compression_no_alpha
        if alpha_disabled == True:
            channels_count = 3

    return channels_count

# Calculate approximate size in MB for NTC compressed image
def ntc_get_size_mb(textures_assets, dims_x, dims_y, bpp):
    largest_texture_id_x = max(enumerate(dims_x),key=lambda x: x[1])[0]
    largest_texture_id_y = max(enumerate(dims_y),key=lambda x: x[1])[0]
    largest_dim_x = dims_x[largest_texture_id_x]
    largest_dim_y = dims_y[largest_texture_id_y]
    largest_texture_id = largest_texture_id_x if largest_dim_x>largest_dim_y else largest_texture_id_y
    
    size_bytes = largest_dim_x * largest_dim_y * bpp / 8
    size_mipchain_bytes = int(size_bytes * 1.0666)

    return size_mipchain_bytes/(1024*1024)

# Get approximate size in MB for BC compressed image
def bc_get_size_mb(textures_asset):
    # modifying texture properties to force load it in editor
    textures_asset.set_editor_property('global_force_mip_levels_to_be_resident',True)
    textures_asset.set_editor_property('never_stream',True)
    texture_mb = textures_asset.blueprint_get_memory_size()/(1024*1024) # not working unless the textures are forced loaded by setting previous properties
    return texture_mb    

# Generate stats, can be exported to CSV file
def ue_generate_stats(ntc_packs, out_csv=False):
    # TODO: Add progress bar as force loading texture assets might be slow
    # TODO: Show confirmation that textures will be dirty but not modified, should not be saved    
    array_pdata = []
    for i in range(len(ntc_packs)):
        package_vram_mb = 0
        package_vram_ntc_mb = 0
        asset_names = []
        asset_sizes = []
        assets_compressmode = []
        asset_dims_x = []
        asset_dims_y = []
        for asset in ntc_packs[i].assets:
            texture = asset.get_asset()
            texture_vram_mb = bc_get_size_mb(texture)
            package_vram_mb += texture_vram_mb
            asset_names.append(asset.asset_name)
            assets_compressmode.append(texture.compression_settings.get_display_name())
            lod_bias = texture.lod_bias
            asset_dims_x.append(texture.blueprint_get_size_x()/(2**lod_bias))
            asset_dims_y.append(texture.blueprint_get_size_y()/(2**lod_bias))
            asset_sizes.append(texture_vram_mb)

        package_vram_ntc_mb = ntc_get_size_mb(ntc_packs[i].assets, asset_dims_x, asset_dims_y, ntc_bpp)
        package_data = NTCPackageStats(ntc_packs[i].name, package_vram_mb, package_vram_ntc_mb, asset_names, assets_compressmode, asset_dims_x, asset_dims_y, asset_sizes, len(asset_names))
        array_pdata.append(package_data)

    if out_csv==True:
        with open(path_csv, 'w') as csvfile:
            data_id=0
            csvfile.write('Package ID, Package, Texture, Mode, DimX, DimY, BC Size (MB), BC Package Size (MB), NTC Package Size(MB), BC/NTC Package Raio\n')
            for p_data in array_pdata:
                data_id+=1
                for i in range(p_data.assets_count):
                    data = [
                        data_id,
                        p_data.package_name,
                        p_data.assets_names[i],
                        str(p_data.assets_compressmode[i]).replace(',',':'),
                        p_data.assets_dims_x[i],
                        p_data.assets_dims_y[i],
                        p_data.assets_sizes[i],
                        p_data.package_size_bc,
                        p_data.package_size_ntc,
                        p_data.package_size_bc/p_data.package_size_ntc,
                        '\n']
                    csvfile.write(','.join(map(str, data)))
        print('Done exporting CSV file')
    else:
        for p_data in array_pdata:
            for i in range(p_data.assets_count):
                print('{} -- Mode {} -- Dims {}x{} -- BC {} MB'.format(
                    filename_getdotdelimt(str(p_data.assets_names[i])),
                    str(p_data.assets_compressmode[i]).replace(',',':'),
                    p_data.assets_dims_x[i],
                    p_data.assets_dims_y[i],
                    round(p_data.assets_sizes[i], 2))
                    )
            print('---- Package {} -- BC {} MB -- NTC {} MB -- Ratio BC/NTC {} -- Count {} textures.'.format(
                p_data.package_name,
                round(p_data.package_size_bc, 2),
                round(p_data.package_size_ntc, 2),
                round((p_data.package_size_bc/p_data.package_size_ntc), 2),
                len(p_data.assets_names)))
            print('-'*64)

# ---------------------
# UE5 Project Scripting
# ---------------------

# City Sample
#ntc_pack_folders = ntc_pack_by_folder(['/Game/Road','/Game/Building/Texture'])
#ntc_pack_names = ntc_pack_by_name(['/Game/Prop','/Game/Vehicle/Texture','/Game/Building/Material','/Game/Building/NY','/Game/Building/SF','/Game/Environment','/Game/Textures/SurfaceFeature'])

ntc_packs = ntc_pack_by_name(['/Game/Prop/Kit_Tree_Maple_Red/Texture'])

ue_generate_stats(ntc_packs, True)
#ue_export_compress_decompress_packs(ntc_packs)
#ue_import_texture_packs(ntc_packs)