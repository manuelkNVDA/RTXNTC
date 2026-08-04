/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <libntc/ntc.h>

/** One semantic role bound to a contiguous slice of the texture's RGBA channels (from manifest \c "semantics"). */
struct ImageSemanticBinding
{
    /** Key inside the per-texture \c "semantics" object (e.g. \c "Albedo", \c "heightmap"). */
    std::string name;
    int firstChannel = 0;
    /** Length of the binding string (\c 1–\c 4): \c R / \c RG / \c RGB / \c RGBA slice starting at \c firstChannel. */
    int numChannels = 0;
};

struct ManifestEntry
{
    std::string fileName;
    /** Raw \c fileName string from manifest JSON (before resolve to absolute). Used when matching optional external
     *  \c semantics.json root keys to manifest rows. */
    std::string manifestJsonFileName;
    std::string entryName;
    std::string channelSwizzle;
    std::vector<ImageSemanticBinding> semantics;
    int mipLevel = 0;
    int firstChannel = -1;
    bool isSRGB = false;
    bool verticalFlip = false;
    ntc::BlockCompressedFormat bcFormat = ntc::BlockCompressedFormat::None;
    std::vector<float> lossFunctionScales;
};

struct Manifest
{
    std::vector<ManifestEntry> textures;
    std::optional<int> width;
    std::optional<int> height;
};

enum class ToolInputType
{
    None,
    Directory,
    CompressedTextureSet,
    ManifestFile,
    ManifestStdin,
    Images,
    Mixed
};

constexpr ntc::BlockCompressedFormat BlockCompressedFormat_Auto = ntc::BlockCompressedFormat(999);

void LowercaseString(std::string& s);
void UppercaseString(std::string& s);

/**
 * Truncate a non-empty \c channelSwizzle (uppercase RGBA letters) to at most \p sourceImageChannelCount entries
 * when semantics implied more channels than the file (e.g. \c RGB placeholder vs BC5 RG normal).
 */
void ClampChannelSwizzleToSourceChannelCount(std::string& channelSwizzle, int sourceImageChannelCount);

std::optional<ntc::BlockCompressedFormat> ParseBlockCompressedFormat(char const* format, bool enableAuto = false);

/** Case-insensitive match for manifest semantic names (ASCII). */
bool ManifestSemanticNameEqualsInsensitive(std::string const& nameUtf8, char const* asciiLiteral);

/** \c true for core alpha-mask aliases: \c AlphaMask, \c Alpha, \c Mask (case-insensitive ASCII). */
bool ManifestSemanticNameIsAlphaMask(std::string const& nameUtf8);

void GenerateManifestFromDirectory(const char* path, bool loadMips, bool keepFileNames, Manifest& outManifest);

void GenerateManifestFromFileList(std::vector<const char*> const& files, bool keepFileNames, Manifest& outManifest);
    
bool ReadManifestFromFile(const char* fileName, Manifest& outManifest, std::string& outError,
    bool ignoreInlineSemantics = false);

bool ReadManifestFromStdin(Manifest& outManifest, std::string& outError);

/**
 * Write manifest JSON. When \p semanticsJsonPathUtf8 is non-null and non-empty, semantics channel strings are taken
 * from that file (same root schema as ReadSemanticsFromFileAndApplyToManifest). When it is null or empty,
 * if a file named \c semantics.json exists beside \p fileName, it is used the same way; otherwise channel strings
 * are derived only from in-memory \c ImageSemanticBinding slice fields (with last-resort defaults when empty).
 */
bool WriteManifestToFile(char const* fileName, Manifest const& manifest, std::string& outError,
    char const* semanticsJsonPathUtf8 = nullptr);

/** Parse manifest JSON (same schema as manifest files) from a null-terminated UTF-8 string. */
bool ReadManifestFromJsonString(char const* jsonUtf8, Manifest& outManifest, std::string& outError,
    bool ignoreInlineSemantics = false);

/**
 * Load semantics bindings from a JSON file and assign them to manifest rows.
 *
 * Each manifest row is matched to a root key in this order (first hit wins). Keys are compared case-insensitively
 * with backslashes normalized to slashes:
 * \c entryName (manifest \c "name"), then the manifest JSON \c fileName string, then the path-shaped id from that
 * JSON path (same rule as the default manifest \c "name" when \c "name" is omitted), then the same id from the
 * resolved absolute \c fileName, then the texture file stem, then the full file basename.
 *
 * File format: UTF-8 JSON object mapping those keys to semantics objects. The same semantics object is applied to
 * every manifest row that resolves to the same key (e.g. all mips), not only the first.
 * Each value is the same object shape as a manifest \c "semantics" property: reserved key \c "isSRGB" (boolean or
 * 0/1 integer) is written only when true; when present and true it sets \c ManifestEntry::isSRGB to true. When
 * absent or false, the manifest row's existing \c isSRGB (from the manifest file) is left unchanged. All other string
 * keys are semantic names mapping to channel binding strings (e.g. \c "RGB", \c "R").
 *
 * Overlay: inline manifest \c semantics are parsed first (including \c \"\" as single-channel \c R). If this file
 * has a matching root key for a row, that row's \c semantics (and derived \c channelSwizzle) are replaced by the
 * file object; otherwise the row keeps its manifest semantics.
 */
bool ReadSemanticsFromFileAndApplyToManifest(char const* semanticsJsonPathUtf8, Manifest& manifest,
    std::string& outError);

bool IsSupportedImageFileExtension(std::string const& extension);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stable name for LoadLibrary/GetProcAddress: parse \p jsonUtf8 and write manifest to \p outputPath.
 * On failure, optional \p errBuf receives a truncated message (UTF-8, null-terminated if errBufChars > 0).
 * Returns 1 on success, 0 on failure.
 */
int NtcUtils_WriteManifestJsonFile(char const* outputPath, char const* jsonUtf8, char* errBuf,
    int errBufChars);

#ifdef __cplusplus
}
#endif

void UpdateToolInputType(ToolInputType& current, ToolInputType newInput);
