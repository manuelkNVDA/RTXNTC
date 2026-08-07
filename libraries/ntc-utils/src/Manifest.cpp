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

#include <ntc-utils/Manifest.h>
#include <filesystem>
#include <json/json.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace
{

/** When JSON omits "name", derive a stable id from the full relative fileName so different folders never
 *  collide on the same stem (e.g. maps/a/normal.png vs maps/b/normal.png were both "normal" before). */
std::string DefaultEntryNameFromRelativeFileName(std::string const& fileNameFromJson)
{
    fs::path const fp(fileNameFromJson);
    std::vector<fs::path> comps;
    for (fs::path const& p : fp)
    {
        if (p.empty() || p == "." || p == "..")
            continue;
        comps.push_back(p);
    }
    if (comps.empty())
        return "texture";

    std::string key;
    for (size_t i = 0; i < comps.size(); ++i)
    {
        if (!key.empty())
            key += '_';
        if (i + 1 == comps.size())
            key += comps[i].stem().generic_string();
        else
            key += comps[i].generic_string();
    }
    return key.empty() ? "texture" : key;
}


/** Only int / uint / double / decimal string count; bool/object/array/null (except missing) → 0 with optional warning.
 *  JsonCpp asInt() maps true→1, false→0, so a mistaken mipLevel:true would skip mip-0 loading for that row. */
int ParseManifestMipLevel(Json::Value const& node, char const* manifestFileNameForLog, std::string const& textureNameForLog)
{
    if (!node.isMember("mipLevel"))
        return 0;
    Json::Value const& m = node["mipLevel"];
    if (m.isNull())
        return 0;

    int v = 0;
    if (m.isInt() || m.isUInt())
        v = m.asInt();
    else if (m.isDouble())
        v = static_cast<int>(m.asDouble());
    else if (m.isString())
    {
        std::string const s = m.asString();
        char* end = nullptr;
        long const n = std::strtol(s.c_str(), &end, 10);
        if (end == s.c_str() || *end != '\0')
        {
            if (manifestFileNameForLog)
                std::fprintf(stderr,
                    "Manifest '%s': texture '%s' has non-integer mipLevel string \"%s\"; using 0.\n",
                    manifestFileNameForLog, textureNameForLog.c_str(), s.c_str());
            return 0;
        }
        v = static_cast<int>(n);
    }
    else
    {
        if (manifestFileNameForLog)
            std::fprintf(stderr,
                "Manifest '%s': texture '%s' has invalid mipLevel (must be a number, not boolean/object); using 0.\n",
                manifestFileNameForLog, textureNameForLog.c_str());
        return 0;
    }

    if (v < 0 || v >= NTC_MAX_MIPS)
    {
        if (manifestFileNameForLog)
            std::fprintf(stderr, "Manifest '%s': texture '%s' mipLevel %d out of range [0,%d); clamping.\n",
                manifestFileNameForLog, textureNameForLog.c_str(), v, NTC_MAX_MIPS);
        if (v < 0)
            v = 0;
        else
            v = NTC_MAX_MIPS - 1;
    }
    return v;
}

} // namespace

void LowercaseString(std::string& s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](uint8_t ch) { return std::tolower(ch); });
}

void UppercaseString(std::string& s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](uint8_t ch) { return std::toupper(ch); });
}

void ClampChannelSwizzleToSourceChannelCount(std::string& channelSwizzle, int sourceImageChannelCount)
{
    if (channelSwizzle.empty() || sourceImageChannelCount <= 0)
        return;
    if ((int)channelSwizzle.size() > sourceImageChannelCount)
        channelSwizzle.resize(static_cast<size_t>(sourceImageChannelCount));
}

bool IsSupportedImageFileExtension(std::string const& extension)
{
    return extension == ".png" ||
           extension == ".jpg" ||
           extension == ".jpeg" ||
           extension == ".tga" ||
           extension == ".exr";
}

static void ComputeDistinctImageNames(Manifest& manifest)
{
    std::string commonName;

    bool isFirstImage = true;
    for (ManifestEntry& entry : manifest.textures)
    {
        // Detect a common prefix for all file names
        if (isFirstImage)
        {
            commonName = entry.entryName;
            isFirstImage = false;
        }
        else
        {
            size_t i;
            for (i = 0; i < commonName.size() && i < entry.entryName.size(); ++i)
            {
                if (tolower(commonName[i]) != tolower(entry.entryName[i]))
                    break;
            }
            commonName.resize(i);
        }
    }

    if (commonName.empty())
        return;

    for (ManifestEntry& entry : manifest.textures)
    {
        std::string distinctName = entry.entryName.substr(commonName.size());
        if (!distinctName.empty())
        {
            distinctName[0] = char(toupper(distinctName[0]));
            entry.entryName = distinctName;
        }
    }
}

void GenerateManifestFromDirectory(const char* path, bool loadMips, bool keepFileNames, Manifest& outManifest)
{
    for (const fs::directory_entry& directoryEntry : fs::directory_iterator(path))
    {
        const fs::path& fileName = directoryEntry.path();

        // Get a lowercase file extension for case-insensitive comparison
        std::string extension = fileName.extension().generic_string();
        LowercaseString(extension);

        if (!IsSupportedImageFileExtension(extension))
            continue;

        ManifestEntry& entry = outManifest.textures.emplace_back();
        entry.fileName = fileName.generic_string();
        entry.manifestJsonFileName = entry.fileName;
        if (keepFileNames)
            entry.entryName = fileName.filename().generic_string();
        else
            entry.entryName = fileName.stem().generic_string();
        entry.mipLevel = 0;
    }

    if (loadMips)
    {
        for (const fs::directory_entry& directoryEntry : fs::directory_iterator(fs::path(path) / "mips"))
        {
            const fs::path& fileName = directoryEntry.path();

            // Get a lowercase file extension for case-insensitive comparison
            std::string extension = fileName.extension().generic_string();
            LowercaseString(extension);

            if (extension != ".png" && extension != ".jpg" && extension != ".tga" && extension != ".exr")
                continue;

            // Parse the file name, assuming it follows this pattern: <name>.<mip>.<type>
            fs::path mip = fileName.stem().extension();
            fs::path name = fileName.stem().stem();

            if (mip.empty() || name.empty())
                continue;

            auto found = std::find_if(outManifest.textures.begin(), outManifest.textures.end(),
                [&name](const ManifestEntry& entry) { return entry.entryName == name; });

            if (found == outManifest.textures.end())
                continue;
            
            int mipLevel = 0;
            if (sscanf(mip.generic_string().c_str(), ".%d", &mipLevel) != 1)
                continue;

            if (mipLevel >= NTC_MAX_MIPS)
                continue;
            
            ManifestEntry& entry = outManifest.textures.emplace_back();
            entry.fileName = fileName.generic_string();
            entry.manifestJsonFileName = entry.fileName;
            entry.entryName = name.generic_string();
            entry.mipLevel = mipLevel;
        }
    }

    if (!keepFileNames)
        ComputeDistinctImageNames(outManifest);
}

void GenerateManifestFromFileList(std::vector<const char *> const &files, bool keepFileNames, Manifest &outManifest)
{
    for (char const* name : files)
    {
        const fs::path& fileName = name;

        ManifestEntry& entry = outManifest.textures.emplace_back();
        entry.fileName = fileName.generic_string();
        entry.manifestJsonFileName = entry.fileName;
        if (keepFileNames)
            entry.entryName = fileName.filename().generic_string();
        else
            entry.entryName = fileName.stem().generic_string();
        entry.mipLevel = 0;
    }
    
    if (!keepFileNames)
        ComputeDistinctImageNames(outManifest);
}

static bool ReadFileIntoVector(FILE* inputFile, std::vector<char>& vector)
{
    if (fseek(inputFile, 0, SEEK_END))
        return false;
    long fileSize = ftell(inputFile);
    if (fseek(inputFile, 0, SEEK_SET))
        return false;
    vector.resize(fileSize);
    if (fread(vector.data(), fileSize, 1, inputFile) != 1)
        return false;
    return true;
}

std::optional<ntc::BlockCompressedFormat> ParseBlockCompressedFormat(char const* format, bool enableAuto)
{
    if (!format || !format[0])
        return ntc::BlockCompressedFormat::None;

    std::string uppercaseFormat = format;
    UppercaseString(uppercaseFormat);

    if (uppercaseFormat == "NONE")
        return ntc::BlockCompressedFormat::None;
    if (uppercaseFormat == "BC1")
        return ntc::BlockCompressedFormat::BC1;
    if (uppercaseFormat == "BC2")
        return ntc::BlockCompressedFormat::BC2;
    if (uppercaseFormat == "BC3")
        return ntc::BlockCompressedFormat::BC3;
    if (uppercaseFormat == "BC4")
        return ntc::BlockCompressedFormat::BC4;
    if (uppercaseFormat == "BC5")
        return ntc::BlockCompressedFormat::BC5;
    if (uppercaseFormat == "BC6" || uppercaseFormat == "BC6H")
        return ntc::BlockCompressedFormat::BC6;
    if (uppercaseFormat == "BC7")
        return ntc::BlockCompressedFormat::BC7;
    if (uppercaseFormat == "AUTO" && enableAuto)
        return BlockCompressedFormat_Auto;

    return std::optional<ntc::BlockCompressedFormat>();
}

std::optional<ntc::ColorSpace> ParseColorSpace(char const* colorSpace)
{
    if (!colorSpace || !colorSpace[0])
        return std::optional<ntc::ColorSpace>();

    std::string uppercaseColorSpace = colorSpace;
    UppercaseString(uppercaseColorSpace);

    if (uppercaseColorSpace == "LINEAR")
        return ntc::ColorSpace::Linear;
    if (uppercaseColorSpace == "SRGB")
        return ntc::ColorSpace::sRGB;
    if (uppercaseColorSpace == "HLG")
        return ntc::ColorSpace::HLG;

    return std::optional<ntc::ColorSpace>();
}

bool ManifestSemanticNameEqualsInsensitive(std::string const& nameUtf8, char const* asciiLiteral)
{
    if (!asciiLiteral)
        return false;
    std::string a = nameUtf8;
    std::string b = asciiLiteral;
    UppercaseString(a);
    UppercaseString(b);
    return a == b;
}

bool ManifestSemanticNameIsAlphaMask(std::string const& nameUtf8)
{
    std::string u = nameUtf8;
    UppercaseString(u);
    return u == "ALPHAMASK" || u == "ALPHA" || u == "MASK";
}

/** Defined below; semantics.json helpers in the anonymous namespace call this before its definition. */
static void NormalizeSemanticsJsonRootKey(std::string& s);

namespace
{
/**
 * idTech / tool manifests often use \c "" as "use default layout for this slot". Matches common idTech
 * material2 / manifest conventions (see also \c ClampChannelSwizzleToSourceChannelCount for BC5 vs RGB).
 */
std::string DefaultChannelsForEmptySemanticKey(std::string const& semanticName)
{
    std::string u = semanticName;
    UppercaseString(u);

    // Gradient helpers (_default placeholders): full RGBA swizzle in manifests.
    if (u.find("GRADIENTMAP") != std::string::npos)
        return "RGBA";

    // BC1-style bloom / screen-space masks: treat as RGB when unspecified.
    if (u == "BLOOMMASKMAP" || u == "BLOOMMASK"
        || (u.find("BLOOM") != std::string::npos && u.find("MASK") != std::string::npos))
        return "RGB";

    // idTech BC1-style SSS / screen-space mask maps (same "" placeholder convention as bloom).
    if (u == "SSSMASK" || u == "SSSMASKMAP"
        || (u.find("SSS") != std::string::npos && u.find("MASK") != std::string::npos))
        return "RGB";

    if (u == "ALBEDO" || u == "BASECOLOR" || u == "DIFFUSE" || u == "COLOR" || u == "SPECULAR"
        || u == "SPECULARCOLOR" || u == "EMISSIVE" || u == "EMISSION" || u == "METALNESS" || u == "METALLIC")
        return "RGB";
    if (u == "NORMAL" || u == "BUMP" || u == "TANGENT")
        return "RGB";
    if (u.size() >= 6 && u.compare(u.size() - 6, 6, "NORMAL") == 0)
        return "RGB";

    // Scalar height / displacement / roughness-style slots (BC4 / single channel in manifests).
    if (u == "HEIGHTMAP" || u == "HEIGHT" || u == "DISPLACEMENT" || u == "ROUGHNESS" || u == "GLOSSINESS"
        || u == "SMOOTHNESS" || u == "OCCLUSION" || u == "AO" || u == "AOMAP" || u == "COVER" || u == "CAVITY"
        || u == "THICKNESS")
        return "R";

    return "R";
}

static std::unordered_map<std::string, Json::Value> BuildSemanticsLookupMap(Json::Value const& root)
{
    std::unordered_map<std::string, Json::Value> out;
    if (!root.isObject())
        return out;
    for (std::string const& texName : root.getMemberNames())
    {
        Json::Value const& semObj = root[texName];
        if (!semObj.isObject())
            continue;
        std::string key = texName;
        NormalizeSemanticsJsonRootKey(key);
        out.insert_or_assign(std::move(key), semObj);
    }
    return out;
}

static Json::Value const* FindSemanticsJsonObjectForEntry(
    std::unordered_map<std::string, Json::Value> const& byNorm, ManifestEntry const& e)
{
    auto const tryKey = [&](std::string key) -> Json::Value const* {
        NormalizeSemanticsJsonRootKey(key);
        auto it = byNorm.find(key);
        if (it == byNorm.end())
            return nullptr;
        return &it->second;
    };
    if (Json::Value const* p = tryKey(e.entryName))
        return p;
    if (!e.manifestJsonFileName.empty())
    {
        if (Json::Value const* p = tryKey(e.manifestJsonFileName))
            return p;
        if (Json::Value const* p = tryKey(DefaultEntryNameFromRelativeFileName(e.manifestJsonFileName)))
            return p;
    }
    if (Json::Value const* p = tryKey(DefaultEntryNameFromRelativeFileName(e.fileName)))
        return p;
    if (Json::Value const* p = tryKey(fs::path(e.fileName).stem().generic_string()))
        return p;
    if (Json::Value const* p = tryKey(fs::path(e.fileName).filename().generic_string()))
        return p;
    return nullptr;
}

static std::string ResolveChannelStringForManifestWrite(Json::Value const* semObj, ImageSemanticBinding const& binding)
{
    static char const* channelMap = "RGBA";
    std::string fromBinding;
    int const count = binding.numChannels;
    if (binding.firstChannel >= 0 && count > 0 && binding.firstChannel + count <= 4)
        fromBinding.assign(channelMap + binding.firstChannel, channelMap + binding.firstChannel + count);

    if (!semObj || !semObj->isObject())
    {
        if (!fromBinding.empty())
            return fromBinding;
        return DefaultChannelsForEmptySemanticKey(binding.name);
    }

    for (std::string const& mem : semObj->getMemberNames())
    {
        if (mem == "isSRGB")
            continue;
        std::string ua = mem;
        std::string ub = binding.name;
        UppercaseString(ua);
        UppercaseString(ub);
        if (ua != ub)
            continue;

        Json::Value const& slotVal = (*semObj)[mem];
        std::string channels;
        if (slotVal.isString())
            channels = slotVal.asString();
        else if (slotVal.isNull())
            channels.clear();
        else if (slotVal.isNumeric() && slotVal.asDouble() == 0.0)
            channels.clear();
        else
            continue;
        UppercaseString(channels);
        if (channels.empty())
            channels = DefaultChannelsForEmptySemanticKey(mem);
        return channels;
    }

    if (!fromBinding.empty())
        return fromBinding;
    return DefaultChannelsForEmptySemanticKey(binding.name);
}

static bool LoadSemanticsJsonFileIntoLookupMap(
    char const* semanticsJsonPathUtf8, std::unordered_map<std::string, Json::Value>& outMap, std::string& outError)
{
    outMap.clear();
    if (!semanticsJsonPathUtf8 || !semanticsJsonPathUtf8[0])
    {
        outError = "LoadSemanticsJsonFileIntoLookupMap: empty path.";
        return false;
    }

    FILE* inputFile = fopen(semanticsJsonPathUtf8, "rb");
    if (!inputFile)
    {
        std::ostringstream oss;
        oss << "Cannot open semantics file '" << semanticsJsonPathUtf8 << "': " << strerror(errno);
        outError = oss.str();
        return false;
    }

    std::vector<char> fileContents;
    bool success = ReadFileIntoVector(inputFile, fileContents);
    fclose(inputFile);
    if (!success)
    {
        std::ostringstream oss;
        oss << "Error while reading semantics file '" << semanticsJsonPathUtf8 << "': " << strerror(errno);
        outError = oss.str();
        return false;
    }

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::CharReader* reader = builder.newCharReader();
    Json::Value root;
    Json::String errorMessages;
    if (!reader->parse(fileContents.data(), fileContents.data() + fileContents.size(), &root, &errorMessages))
    {
        std::ostringstream oss;
        oss << "Cannot parse semantics file '" << semanticsJsonPathUtf8 << "': " << errorMessages;
        outError = oss.str();
        return false;
    }

    if (!root.isObject())
    {
        outError = "Malformed semantics file: root must be a JSON object mapping texture names to semantics objects.";
        return false;
    }

    outMap = BuildSemanticsLookupMap(root);
    return true;
}
} // namespace

static bool ParseSemanticsJsonObjectIntoBindings(Json::Value const& semanticsNode, char const* textureNameForErrors,
    std::vector<ImageSemanticBinding>& outSemantics, std::optional<bool>& outIsSRGBFromSemantics,
    std::string& outError)
{
    if (!semanticsNode.isObject())
    {
        outError = "Semantics value must be a JSON object.";
        return false;
    }
    outSemantics.clear();
    outIsSRGBFromSemantics.reset();
    for (std::string const& semanticName : semanticsNode.getMemberNames())
    {
        if (semanticName.empty())
        {
            outError = "Semantics object contains an empty semantic name key.";
            return false;
        }

        if (semanticName == "isSRGB")
        {
            Json::Value const& v = semanticsNode[semanticName];
            bool b = false;
            if (v.isBool())
                b = v.asBool();
            else if (v.isInt() || v.isUInt())
                b = (v.asInt() != 0);
            else
            {
                std::ostringstream oss;
                oss << "Invalid 'isSRGB' for texture '" << textureNameForErrors << "' (must be boolean or 0/1).";
                outError = oss.str();
                return false;
            }
            if (b)
                outIsSRGBFromSemantics = true;
            continue;
        }

        Json::Value const& slotVal = semanticsNode[semanticName];
        std::string channels;
        if (slotVal.isString())
            channels = slotVal.asString();
        else if (slotVal.isNull())
            channels.clear();
        else if (slotVal.isNumeric() && slotVal.asDouble() == 0.0)
            channels.clear(); // some exporters use 0 instead of "" for "use default layout"
        else
        {
            std::ostringstream oss;
            oss << "Texture '" << textureNameForErrors << "': semantics slot '" << semanticName
                << "' must be a string (R, RG, RGB, RGBA, or empty for defaults), null, or numeric 0.";
            outError = oss.str();
            return false;
        }
        UppercaseString(channels);
        // idTech-style "" / null / 0 = default channel layout for this semantic slot name (not always scalar R).
        if (channels.empty())
            channels = DefaultChannelsForEmptySemanticKey(semanticName);
        static char const* channelMap = "RGBA";
        char const* firstChannelPtr = strstr(channelMap, channels.c_str());
        if (!firstChannelPtr)
        {
            std::ostringstream oss;
            oss << "Invalid semantic binding '" << channels << "' for texture '" << textureNameForErrors << "' semantic '"
                << semanticName << "'. Semantic bindings must use sequential channels from RGBA set.";
            outError = oss.str();
            return false;
        }

        int const numCh = int(channels.size());
        if (numCh < 1 || numCh > 4)
        {
            outError = "Semantic binding must use 1 to 4 channel letters (R, RG, RGB, or RGBA).";
            return false;
        }

        ImageSemanticBinding binding;
        binding.name = semanticName;
        binding.firstChannel = int(firstChannelPtr - channelMap);
        binding.numChannels = numCh;
        outSemantics.push_back(binding);
    }
    return true;
}

static bool DeriveChannelSwizzleFromSemanticsIfEmpty(ManifestEntry& entry, std::string& outError)
{
    if (!entry.channelSwizzle.empty() || entry.semantics.empty())
        return true;

    static constexpr char kRGBA[] = "RGBA";
    std::vector<size_t> ord(entry.semantics.size());
    for (size_t i = 0; i < ord.size(); ++i)
        ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
        return entry.semantics[a].firstChannel < entry.semantics[b].firstChannel;
    });
    std::string autoSwizzle;
    for (size_t const bi : ord)
    {
        ImageSemanticBinding const& binding = entry.semantics[bi];
        int const n = binding.numChannels;
        if (n <= 0 || binding.firstChannel < 0 || binding.firstChannel + n > 4)
        {
            std::ostringstream oss;
            oss << "Cannot derive channelSwizzle for texture '" << entry.entryName << "'.";
            outError = oss.str();
            return false;
        }
        autoSwizzle.append(kRGBA + binding.firstChannel, static_cast<size_t>(n));
    }
    entry.channelSwizzle = std::move(autoSwizzle);
    return true;
}

static bool ParseManifest(char const* jsonData, size_t jsonSize, char const* manifestFileName, Manifest& outManifest,
    std::string& outError, bool ignoreInlineSemantics)
{
    std::unordered_set<std::string> mip0EntryNames;
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::CharReader* reader = builder.newCharReader();

    Json::Value root;
    Json::String errorMessages;
    if (!reader->parse(jsonData, jsonData + jsonSize, &root, &errorMessages))
    {
        std::ostringstream oss;
        if (!manifestFileName)
            oss << "Cannot parse the manifest: " << errorMessages;
        else
            oss << "Cannot parse manifest file '" << manifestFileName << "': " << errorMessages;
        outError = oss.str();
        return false;
    }

    if (!root.isObject() && !root.isArray())
    {
        outError = "Malformed manifest: document root must be an object or an array.";
        return false;
    }

    // Select between the new format `{ "textures": [...] }` and the old format `[...]` for compatibility.
    // The old format will be removed someday.
    Json::Value const& textures = root.isObject() ? root["textures"] : root;
    if (!textures.isArray() || textures.empty())
    {
        outError = "Malformed manifest: must contain a non-empty 'textures' array.";
        return false;
    }

    if (root.isObject())
    {
        if (root["width"].isNumeric())
            outManifest.width = root["width"].asInt();
        if (root["height"].isNumeric())
            outManifest.height = root["height"].asInt();
    }

    for (const auto& node: textures)
    {
        if (!node.isObject())
        {
            outError = "Malformed manifest: all entries in the textures array must be objects.";
            return false;
        }

        std::string const fileName = node["fileName"].asString();
        if (fileName.empty())
        {
            outError = "Malformed manifest: texture entry has empty or missing 'fileName'.";
            return false;
        }

        ManifestEntry entry;
        entry.manifestJsonFileName = fileName;
        if (manifestFileName)
        {
            // Relative paths are resolved against the manifest file's directory. Absolute paths in JSON
            // are used as explicit locations (no join), so tools can reference assets outside that tree.
            fs::path const texturePath(fileName);
            if (texturePath.is_absolute())
                entry.fileName = texturePath.lexically_normal().generic_string();
            else
                entry.fileName = (fs::path(manifestFileName).parent_path() / texturePath).lexically_normal().generic_string();
        }
        else
            entry.fileName = fileName;
        entry.entryName = node["name"].asString();
        if (entry.entryName.empty())
            entry.entryName = DefaultEntryNameFromRelativeFileName(fileName);
        entry.mipLevel = ParseManifestMipLevel(node, manifestFileName, entry.entryName);
        entry.isSRGB = node["isSRGB"].asBool();
        entry.verticalFlip = node["verticalFlip"].asBool();
        entry.channelSwizzle = node["channelSwizzle"].asString();
        auto firstChannel = node["firstChannel"];
        entry.firstChannel = firstChannel.isInt() ? firstChannel.asInt() : entry.firstChannel;

        // Normalize and validate the channel selection
        if (!entry.channelSwizzle.empty())
        {
            UppercaseString(entry.channelSwizzle);
            bool valid = true;

            if (entry.channelSwizzle.size() > 4)
                valid = false;

            for (char c : entry.channelSwizzle)
            {
                if (!strchr("RGBA", c))
                    valid = false;
            }

            if (!valid)
            {
                std::ostringstream oss;
                oss << "Invalid channel swizzle '" << entry.channelSwizzle << "' specified for texture '"
                    << entry.entryName << "'. It must be 0-4 characters long and contain only RGBA characters.";
                outError = oss.str();
                return false;
            }
        }
        
        // Parse the output format
        std::string bcFormat = node["bcFormat"].asString();
        if (bcFormat.empty())
            bcFormat = node["outputFormat"].asString(); // Legacy version
        if (!bcFormat.empty())
        {
            auto parsedFormat = ParseBlockCompressedFormat(bcFormat.c_str(), true);
            if (parsedFormat.has_value())
            {
                entry.bcFormat = parsedFormat.value();
            }
            else
            {
                std::ostringstream oss;
                oss << "Unknown format '" << bcFormat.c_str() << "' specified for texture '" << entry.entryName << "'.";
                outError = oss.str();
                return false;
            }
        }

        // Parse the storage color space override
        std::string const storageColorSpace = node["storageColorSpace"].asString();
        if (!storageColorSpace.empty())
        {
            entry.storageColorSpace = ParseColorSpace(storageColorSpace.c_str());
            if (!entry.storageColorSpace.has_value())
            {
                std::ostringstream oss;
                oss << "Unknown storage color space '" << storageColorSpace << "' specified for texture '"
                    << entry.entryName << "'. Must be one of: linear, srgb, hlg.";
                outError = oss.str();
                return false;
            }
        }

        // Parse per-texture semantic bindings from manifest (skipped when ignoreInlineSemantics is set)
        if (!ignoreInlineSemantics)
        {
            Json::Value const& semanticsNode = node["semantics"];
            if (semanticsNode.isObject())
            {
                std::optional<bool> semanticsIsSRGB;
                if (!ParseSemanticsJsonObjectIntoBindings(semanticsNode, entry.entryName.c_str(), entry.semantics,
                        semanticsIsSRGB, outError))
                    return false;
                if (semanticsIsSRGB.has_value())
                    entry.isSRGB = true;
            }
            else if (!semanticsNode.isNull())
            {
                outError = "Malformed manifest: 'semantics' property must be an object.";
                return false;
            }

            if (!DeriveChannelSwizzleFromSemanticsIfEmpty(entry, outError))
                return false;
        }

        Json::Value const& lossFunctionScaleNode = node["lossFunctionScale"];
        if (lossFunctionScaleNode.isArray())
        {
            for (const auto& scaleNode : lossFunctionScaleNode)
            {
                if (!scaleNode.isNumeric())
                {
                    outError = "Malformed manifest: all entries in the 'lossFunctionScale' array must be numeric.";
                    return false;
                }
                entry.lossFunctionScales.push_back(scaleNode.asFloat());
            }
        }
        else if (lossFunctionScaleNode.isNumeric())
        {
            entry.lossFunctionScales.push_back(lossFunctionScaleNode.asFloat());
        }
        else if (!lossFunctionScaleNode.isNull())
        {
            outError = "Malformed manifest: 'lossFunctionScale' property must be a number or an array of numbers.";
            return false;
        }

        if (entry.mipLevel == 0)
        {
            auto const inserted = mip0EntryNames.insert(entry.entryName);
            if (!inserted.second)
            {
                outError = "Malformed manifest: duplicate mip-0 texture name '" + entry.entryName +
                    "'. Use a unique \"name\" per map, or omit \"name\" so it is derived from the full "
                    "\"fileName\" path (folder + file).";
                return false;
            }
        }

        outManifest.textures.push_back(entry);
    }

    return true;
}

bool ReadManifestFromFile(const char* fileName, Manifest& outManifest, std::string& outError,
    bool ignoreInlineSemantics)
{
    FILE* inputFile = fopen(fileName, "rb");
    if (!inputFile)
    {
        std::ostringstream oss;
        oss << "Cannot open manifest file '" << fileName << "': " << strerror(errno);
        outError = oss.str();
        return false;
    }

    std::vector<char> fileContents;
    bool success = ReadFileIntoVector(inputFile, fileContents);
    fclose(inputFile);
    
    if (!success)
    {
        std::ostringstream oss;
        oss << "Error while reading manifest file '" << fileName << "': " << strerror(errno);
        outError = oss.str();
        return false;
    }

    return ParseManifest(fileContents.data(), fileContents.size(), fileName, outManifest, outError,
        ignoreInlineSemantics);
}

bool ReadManifestFromStdin(Manifest& outManifest, std::string& outError)
{
    std::stringstream ss;
    std::string line;
    while (std::getline(std::cin, line))
    {
        ss << line;
        if (!std::cin.eof())
            ss << std::endl;
    }

    std::string const inputString = ss.str();

    return ParseManifest(inputString.data(), inputString.size(), nullptr, outManifest, outError, false);
}

bool WriteManifestToFile(char const* fileName, Manifest const& manifest, std::string& outError,
    char const* semanticsJsonPathUtf8)
{
    fs::path const manifestDir = fs::absolute(fileName).parent_path();

    std::string semanticsPath;
    if (semanticsJsonPathUtf8 && semanticsJsonPathUtf8[0])
        semanticsPath = semanticsJsonPathUtf8;
    else
    {
        fs::path const sidecar = manifestDir / "semantics.json";
        std::error_code ec;
        if (fs::is_regular_file(sidecar, ec))
            semanticsPath = sidecar.generic_string();
    }

    std::optional<std::unordered_map<std::string, Json::Value>> semanticsByNorm;
    if (!semanticsPath.empty())
    {
        std::unordered_map<std::string, Json::Value> loaded;
        if (!LoadSemanticsJsonFileIntoLookupMap(semanticsPath.c_str(), loaded, outError))
            return false;
        semanticsByNorm = std::move(loaded);
    }

    Json::Value root;
    if (manifest.width.has_value())
        root["width"] = manifest.width.value();
    if (manifest.height.has_value())
        root["height"] = manifest.height.value();
    Json::Value textures(Json::arrayValue);

    for (const ManifestEntry& entry : manifest.textures)
    {
        // Update the texture file name to be relative to the new manifest file location
        std::string textureFileName = entry.fileName;
        if (!textureFileName.empty())
        {
            fs::path const texturePath = fs::absolute(textureFileName);
            std::error_code ec;
            fs::path relativePath = fs::relative(texturePath, manifestDir, ec);
            if (!ec && !relativePath.empty())
            {
                textureFileName = relativePath.generic_string();
            }
        }
        
        Json::Value node;
        node["fileName"] = textureFileName;
        node["name"] = entry.entryName;
        node["isSRGB"] = entry.isSRGB;
        if (entry.mipLevel > 0)
            node["mipLevel"] = entry.mipLevel;
        if (entry.verticalFlip)
            node["verticalFlip"] = entry.verticalFlip;
        if (!entry.channelSwizzle.empty())
            node["channelSwizzle"] = entry.channelSwizzle;
        if (entry.firstChannel >= 0)
            node["firstChannel"] = entry.firstChannel;

        // BC format
        switch(entry.bcFormat)
        {
            case BlockCompressedFormat_Auto:
                // "auto" is a custom value used in the SDK, not part of the official enum
                node["bcFormat"] = "auto";
                break;
            case ntc::BlockCompressedFormat::None:
                // "none" is the default, no need to write it out
                break;
            default:
                node["bcFormat"] = ntc::BlockCompressedFormatToString(entry.bcFormat);
                break;
        }

        // Storage color space is only written when it was requested explicitly, so that omitting it keeps
        // the format-derived default.
        if (entry.storageColorSpace.has_value())
            node["storageColorSpace"] = ntc::ColorSpaceToString(entry.storageColorSpace.value());

        // Semantics: channel bindings only; sRGB is expressed only as top-level "isSRGB" (not duplicated here).
        if (!entry.semantics.empty())
        {
            Json::Value semanticsNode(Json::objectValue);
            Json::Value const* semForEntry = nullptr;
            if (semanticsByNorm.has_value())
                semForEntry = FindSemanticsJsonObjectForEntry(*semanticsByNorm, entry);
            for (const auto& binding : entry.semantics)
            {
                std::string const channels = ResolveChannelStringForManifestWrite(semForEntry, binding);
                semanticsNode[binding.name] = channels;
            }
            node["semantics"] = semanticsNode;
        }

        // Loss function scales
        if (!entry.lossFunctionScales.empty())
        {
            if (entry.lossFunctionScales.size() == 1)
                node["lossFunctionScale"] = entry.lossFunctionScales[0];
            else
            {
                Json::Value arr(Json::arrayValue);
                for (float v : entry.lossFunctionScales)
                    arr.append(v);
                node["lossFunctionScale"] = arr;
            }
        }

        textures.append(node);
    }

    root["textures"] = textures;

    Json::StreamWriterBuilder writerBuilder;
    writerBuilder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(writerBuilder.newStreamWriter());

    std::ofstream ofs(fileName, std::ios::out | std::ios::trunc);
    if (!ofs)
    {
        std::ostringstream oss;
        oss << "Cannot open manifest file '" << fileName << "' for writing: " << strerror(errno);
        outError = oss.str();
        return false;
    }

    // JsonCpp StreamWriter::write always returns 0; failures show up on the stream (see json.h on write()).
    writer->write(root, &ofs);
    ofs.flush();
    if (!ofs.good())
    {
        outError = "Failed to write manifest JSON to file.";
        return false;
    }

    return true;
}

bool ReadManifestFromJsonString(char const* jsonUtf8, Manifest& outManifest, std::string& outError,
    bool ignoreInlineSemantics)
{
    if (!jsonUtf8)
    {
        outError = "ReadManifestFromJsonString: null jsonUtf8";
        return false;
    }
    size_t const len = std::strlen(jsonUtf8);
    return ParseManifest(jsonUtf8, len, nullptr, outManifest, outError, ignoreInlineSemantics);
}

/** Lowercase + unify slashes so manifest \c "name" / paths match \c semantics.json root keys from Windows or Unix. */
static void NormalizeSemanticsJsonRootKey(std::string& s)
{
    LowercaseString(s);
    for (char& c : s)
    {
        if (c == '\\')
            c = '/';
    }
}

bool ReadSemanticsFromFileAndApplyToManifest(char const* semanticsJsonPathUtf8, Manifest& manifest,
    std::string& outError)
{
    if (!semanticsJsonPathUtf8 || !semanticsJsonPathUtf8[0])
    {
        outError = "ReadSemanticsFromFileAndApplyToManifest: empty path.";
        return false;
    }

    FILE* inputFile = fopen(semanticsJsonPathUtf8, "rb");
    if (!inputFile)
    {
        std::ostringstream oss;
        oss << "Cannot open semantics file '" << semanticsJsonPathUtf8 << "': " << strerror(errno);
        outError = oss.str();
        return false;
    }

    std::vector<char> fileContents;
    bool success = ReadFileIntoVector(inputFile, fileContents);
    fclose(inputFile);
    if (!success)
    {
        std::ostringstream oss;
        oss << "Error while reading semantics file '" << semanticsJsonPathUtf8 << "': " << strerror(errno);
        outError = oss.str();
        return false;
    }

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::CharReader* reader = builder.newCharReader();
    Json::Value root;
    Json::String errorMessages;
    if (!reader->parse(fileContents.data(), fileContents.data() + fileContents.size(), &root, &errorMessages))
    {
        std::ostringstream oss;
        oss << "Cannot parse semantics file '" << semanticsJsonPathUtf8 << "': " << errorMessages;
        outError = oss.str();
        return false;
    }

    if (!root.isObject())
    {
        outError = "Malformed semantics file: root must be a JSON object mapping texture names to semantics objects.";
        return false;
    }

    // Root keys normalized (case + slashes) so manifest "name" and fileName strings match semantics.json.
    // Bindings are copied onto each manifest row (never moved out of the map): multiple rows share the
    // same entryName (e.g. mip0/mip1) and must all receive the same semantics.
    std::unordered_map<std::string, std::pair<std::vector<ImageSemanticBinding>, std::optional<bool>>> byName;
    for (std::string const& texName : root.getMemberNames())
    {
        Json::Value const& semObj = root[texName];
        if (!semObj.isObject())
            continue;
        std::vector<ImageSemanticBinding> bindings;
        std::optional<bool> isSRGBFromSemantics;
        if (!ParseSemanticsJsonObjectIntoBindings(semObj, texName.c_str(), bindings, isSRGBFromSemantics, outError))
            return false;
        std::string key = texName;
        NormalizeSemanticsJsonRootKey(key);
        byName[std::move(key)] = std::make_pair(std::move(bindings), std::move(isSRGBFromSemantics));
    }

    using SemanticsMapIt = std::unordered_map<std::string,
        std::pair<std::vector<ImageSemanticBinding>, std::optional<bool>>>::iterator;
    auto findByNormalizedKey = [&](std::string key) -> SemanticsMapIt {
        NormalizeSemanticsJsonRootKey(key);
        return byName.find(key);
    };

    auto findSemanticsForEntry = [&](ManifestEntry const& e) -> SemanticsMapIt {
        SemanticsMapIt it = findByNormalizedKey(e.entryName);
        if (it != byName.end())
            return it;
        if (!e.manifestJsonFileName.empty())
        {
            it = findByNormalizedKey(e.manifestJsonFileName);
            if (it != byName.end())
                return it;
            it = findByNormalizedKey(DefaultEntryNameFromRelativeFileName(e.manifestJsonFileName));
            if (it != byName.end())
                return it;
        }
        it = findByNormalizedKey(DefaultEntryNameFromRelativeFileName(e.fileName));
        if (it != byName.end())
            return it;
        it = findByNormalizedKey(fs::path(e.fileName).stem().generic_string());
        if (it != byName.end())
            return it;
        return findByNormalizedKey(fs::path(e.fileName).filename().generic_string());
    };

    for (ManifestEntry& e : manifest.textures)
    {
        auto it = findSemanticsForEntry(e);
        if (it == byName.end())
        {
            // No file entry for this texture: keep manifest-embedded semantics (already parsed).
            continue;
        }
        e.semantics = it->second.first;
        if (it->second.second.has_value())
            e.isSRGB = true;
        if (!DeriveChannelSwizzleFromSemanticsIfEmpty(e, outError))
            return false;
    }

    return true;
}

static void CopyErrorToBuffer(std::string const& err, char* errBuf, int errBufChars)
{
    if (!errBuf || errBufChars <= 0)
        return;
    std::strncpy(errBuf, err.c_str(), static_cast<size_t>(errBufChars) - 1U);
    errBuf[errBufChars - 1] = '\0';
}

extern "C" int NtcUtils_WriteManifestJsonFile(char const* outputPath, char const* jsonUtf8, char* errBuf,
    int errBufChars)
{
    if (!outputPath || !jsonUtf8)
    {
        CopyErrorToBuffer("NtcUtils_WriteManifestJsonFile: null path or json", errBuf, errBufChars);
        return 0;
    }
    Manifest manifest;
    std::string err;
    if (!ReadManifestFromJsonString(jsonUtf8, manifest, err))
    {
        CopyErrorToBuffer(err, errBuf, errBufChars);
        return 0;
    }
    if (!WriteManifestToFile(outputPath, manifest, err))
    {
        CopyErrorToBuffer(err, errBuf, errBufChars);
        return 0;
    }
    return 1;
}

void UpdateToolInputType(ToolInputType& current, ToolInputType newInput)
{
    switch(current)
    {
        case ToolInputType::None:
            // First input, use its type
            current = newInput;
            return;
        case ToolInputType::Directory:
        case ToolInputType::CompressedTextureSet:
        case ToolInputType::ManifestFile:
        case ToolInputType::ManifestStdin:
            // Mismatching input types or using more than one of these is not allowed
            current = ToolInputType::Mixed;
            return;
        case ToolInputType::Images:
            // Multiple images are allowed, mixing images with other types is not
            if (newInput != ToolInputType::Images)
                current = ToolInputType::Mixed;
            return;
    }
}
