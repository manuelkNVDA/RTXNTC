/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <ntc-utils/Semantics.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{
char const kNtcSemMarker[] = "|ntcsem:";
}

std::string BuildTextureNameWithEmbeddedSemantics(std::string const& baseName,
    std::vector<ImageSemanticBinding> const& semantics)
{
    if (semantics.empty())
        return baseName;
    std::string s = baseName;
    s += kNtcSemMarker;
    for (size_t i = 0; i < semantics.size(); ++i)
    {
        if (i > 0)
            s += ',';
        s += semantics[i].name;
        s += ':';
        s += std::to_string(semantics[i].firstChannel);
        s += ':';
        s += std::to_string(semantics[i].numChannels);
    }
    return s;
}

std::string StripNtcSemanticsSuffixForDisplay(std::string const& textureName)
{
    std::string::size_type const pos = textureName.find(kNtcSemMarker);
    if (pos == std::string::npos)
        return textureName;
    return textureName.substr(0, pos);
}

void GuessImageSemantics(std::string const& distinctName, int channels, ntc::ChannelFormat channelFormat,
    int imageIndex, bool& outIsSRGB, std::vector<SemanticBinding>& outSemantics)
{
    (void)channels;
    (void)channelFormat;
    (void)outIsSRGB;

    std::string::size_type const markerPos = distinctName.find(kNtcSemMarker);
    if (markerPos == std::string::npos)
        return;

    std::string const tagRegion = distinctName.substr(markerPos + sizeof(kNtcSemMarker) - 1);
    std::vector<SemanticBinding> parsed;
    for (size_t at = 0; at < tagRegion.size();)
    {
        size_t const comma = tagRegion.find(',', at);
        std::string const token = tagRegion.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? tagRegion.size() : comma + 1;

        size_t const colon = token.find(':');
        if (colon == std::string::npos)
            continue;
        size_t const colon2 = token.find(':', colon + 1);
        std::string const lab = token.substr(0, colon);
        int const fc = std::atoi(token.c_str() + colon + 1);
        int nc = 4;
        if (colon2 != std::string::npos)
            nc = std::atoi(token.c_str() + colon2 + 1);
        if (nc < 1)
            nc = 1;
        if (nc > 4)
            nc = 4;
        if (lab.empty())
            continue;
        parsed.push_back({ lab, imageIndex, fc, nc });
    }

    if (parsed.empty())
        return;

    outSemantics.insert(outSemantics.end(), parsed.begin(), parsed.end());
}
