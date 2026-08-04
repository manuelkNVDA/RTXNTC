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

#pragma once

#include <ntc-utils/Manifest.h>
#include <string>
#include <vector>

struct SemanticBinding
{
    /** Semantic name (same convention as manifest / semantics JSON keys). */
    std::string name;
    int imageIndex = 0;
    int firstChannel = 0;
    /** Channel span for this binding (1–4), used when mapping to packed layouts / UI. */
    int numChannels = 1;
};

/** Optional: parse \c "|ntcsem:Name:firstCh:numCh,..." embedded in a texture display name. Does not infer
 *  semantics from paths or filenames — use manifest \c "semantics" (and an optional \c semantics.json overlay when
 *  integrated by the host) for authoritative bindings and \c isSRGB. \p outIsSRGB is not modified (reserved for API stability). */
void GuessImageSemantics(std::string const& distinctName, int channels, ntc::ChannelFormat channelFormat,
    int imageIndex, bool& outIsSRGB, std::vector<SemanticBinding>& outSemantics);

/** Appends "|ntcsem:Label:firstCh,..." so viewers can recover manifest semantics from a standalone .ntc. */
std::string BuildTextureNameWithEmbeddedSemantics(std::string const& baseName,
    std::vector<ImageSemanticBinding> const& semantics);

/** Removes embedded semantics suffix for UI labels (see BuildTextureNameWithEmbeddedSemantics). */
std::string StripNtcSemanticsSuffixForDisplay(std::string const& textureName);
