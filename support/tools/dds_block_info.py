#!/usr/bin/python

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import struct
import argparse

class DDSHeader:
    pass

def readDDSHeader(file):
    header_format = '4s 20I 4s 10I'
    header_size = struct.calcsize(header_format)

    header_data = file.read(header_size)
    header = struct.unpack(header_format, header_data)
    
    if header[0] != b'DDS ':
        raise ValueError("Not a valid DDS file")

    h = DDSHeader()
    h.width = header[4]
    h.height = header[3]
    if header[21] != b'DX10':
        raise ValueError("DX10 header not found")

    dx10header_format = '5I'
    dx10header_size = struct.calcsize(dx10header_format)
    dx10header_data = file.read(dx10header_size)
    dx10header = struct.unpack(dx10header_format, dx10header_data)
    dxgiFormat = dx10header[0]
    if dxgiFormat < 97 or dxgiFormat > 99:
        raise ValueError(f"DXGI format = {dxgiFormat}, expected BC7 (97-99)")

    h.dataOffset = header_size + dx10header_size

    return h

def readBC7Block(file, header, x, y):
    bytesPerBlock = 16  # Size of a BC7 block in bytes
    widthInBlocks = (header.width + 3) // 4
    blockOffset = (y * widthInBlocks + x) * bytesPerBlock + header.dataOffset

    file.seek(blockOffset)
    block_data = file.read(bytesPerBlock)

    return block_data

# Returns the index of the lowest '1' bit in an integer
def ffs(x):
    return (x & -x).bit_length() - 1

def extractBits(x, offset, length):
    return (x >> offset) & ((1 << length) - 1)

def expandBits(q, bits):
    return (q << (8 - bits)) | (q >> (2 * bits - 8))

def getEndpoint(blockValue, offset, bits, stride, channels, pBitOffset):
    epValues = [extractBits(blockValue, offset + ch * stride, bits) for ch in range(channels)]
    if pBitOffset is not None:
        pBit = extractBits(blockValue, pBitOffset, 1)
        epValues = [(x << 1) | pBit for x in epValues]
        bits += 1
    return [expandBits(x, bits) for x in epValues]
    
def printBC7Block(block_data):
    value = int.from_bytes(block_data, "little")
    mode = min(7, ffs(value))
    partitionMask = [ 15, 63, 63, 63, 7, 3, 0, 63 ]
    partition = (value >> (mode + 1)) & partitionMask[mode]

    print(f"Mode = {mode}, partition = {partition}")

    endpoints = []
    if mode == 0:
        endpoints = [getEndpoint(value, 5 + i * 4, 4, 24, 3, 77 + i) for i in range(6)]
    elif mode == 1:
        endpoints = [getEndpoint(value, 8 + i * 6, 6, 24, 3, 80 + i) for i in range(4)]
    elif mode == 2:
        endpoints = [getEndpoint(value, 9 + i * 5, 5, 30, 3, None) for i in range(6)]
    elif mode == 3:
        endpoints = [getEndpoint(value, 10 + i * 7, 7, 28, 3, 94 + i) for i in range(4)]
    elif mode == 4:
        endpoints = [getEndpoint(value, 8 + i * 5, 5, 10, 3, None) for i in range(2)]
        endpoints[0].append(getEndpoint(value, 38, 6, 0, 1, None)[0]) # A0
        endpoints[1].append(getEndpoint(value, 44, 6, 0, 1, None)[0]) # A1
    elif mode == 5:
        endpoints = [getEndpoint(value, 8 + i * 7, 7, 14, 3, None) for i in range(2)]
        endpoints[0].append(getEndpoint(value, 50, 8, 0, 1, None)[0]) # A0
        endpoints[1].append(getEndpoint(value, 58, 8, 0, 1, None)[0]) # A1
    elif mode == 6:
        endpoints = [getEndpoint(value, 7 + i * 7, 7, 14, 4, 63 + i) for i in range(4)]
    elif mode == 7:
        endpoints = [getEndpoint(value, 14 + i * 5, 5, 20, 4, 94 + i) for i in range(4)]

    for i in range(len(endpoints)):
        print(f"EP[{i}] = ({', '.join([str(x) for x in endpoints[i]])})")

def main():
    parser = argparse.ArgumentParser(description='Print information about a BC7 block in a DDS file.')
    parser.add_argument('dds_file', help='Path to the DDS file')
    parser.add_argument('x', type=int, help='X coordinate of the pixel')
    parser.add_argument('y', type=int, help='Y coordinate of the pixel')
    args = parser.parse_args()

    try:
        with open(args.dds_file, "rb") as file:
            header = readDDSHeader(file)
            print(f"DDS Header: {header.__dict__}")

            blockX = args.x // 4
            blockY = args.y // 4
            block_data = readBC7Block(file, header, blockX, blockY)

        printBC7Block(block_data)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
