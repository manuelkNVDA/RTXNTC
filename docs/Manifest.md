# NTC Manifest File Format

Manifest files are using the JSON format with the following schema.

## Document

The root of the manifest document is an object with the following fields.

| Field Name       | Type                     | Default    | Description 
|------------------|--------------------------|------------|-------------
| `textures`       | array of `ManifestEntry` | (required) | List of textures to include in the texture set
| `width`          | int                      | derived    | Custom width for the texture set
| `height`         | int                      | derived    | Custom height for the texture set

## `ManifestEntry` object

| Field Name          | Type           | Default    | Description 
|---------------------|----------------|------------|-------------
| `fileName`          | string         | (required) | Path to the texture image file relative to the manifest file location.
| `bcFormat`          | string         | `none`     | Block compression format (`BC1` - `BC7`) that should be used for transcoding of this texture after NTC decompression. This is only a hint, and implementations may use a different format.
| `channelSwizzle`    | string         | derived    | Set and order of channels from this image that will be used in the NTC texture set. Must be 1-4 characters long and only contain `R, G, B, A` characters, such as `"BGR"`. If not specified, all channels from the image are used in their original order.
| `firstChannel`      | int            | derived    | First channel in the NTC texture set that will be occupied by this texture, 0-15. If not specified, the first available channel is selected. The texture's channels (after swizzle) must fit into the texture set, i.e. no channel may have an index higher than 15.
| `isSRGB`            | bool           | `false`    | If set to `true`, the texture data will be interpreted as sRGB encoded.
| `lossFunctionScale` | float or array | 1.0 | Loss function scales for the texture's channels as an array, or a single number to apply to all channels.
| `mipLevel`          | int            | 0          | Mip level in the NTC texture set that the specified image will be loaded into.
| `name`              | string         | derived    | Name of the texture in the NTC texture set. If not provided, the file name without extension is used instead.
| `semantics`         | object         | empty      | Map of semantic labels (strings) to channel ranges (also strings). For each semantic label (see below for a full list), a set of channels is specified using a substring of `"RGBA"`. This means that channel order must be preserved; `"RG"` and `"GBA"` are valid channel ranges, while `"BGA"` is not.
| `storageColorSpace` | string         | derived    | Color space that the texture's channels are quantized and stored in: `linear`, `srgb` or `hlg`. See below.
| `verticalFlip`      | bool           | `false`    | If set to `true`, the texture will be flipped along the Y axis on load.

## Storage color space

`isSRGB` describes how the *source image* is encoded. `storageColorSpace` describes the transfer function that the
data is quantized in *inside the NTC file*, which is what determines where the compressor spends its precision.
The two are independent: the encoder converts from one to the other on load and back on decompression, so the
round trip is correct for any combination.

When `storageColorSpace` is not specified, the default depends on the source image's channel format:

- Floating point images (`.exr`) are stored as `hlg`.
- All other images are stored in their source color space, i.e. `srgb` when `isSRGB` is set and `linear` otherwise.

Hybrid Log-Gamma is a good default for HDR *color*, where it maps an unbounded range into a compact domain and
distributes quantization levels perceptually. It is a poor fit for floating point channels that do not carry color,
such as displacement, height, curvature, world-space positions or signed distance fields. HLG concentrates
quantization levels around zero without bound and starves values well above 1.0, so for a height field it spends
its resolution on the arbitrary zero of the bake rather than on the tallest features. Set `storageColorSpace` to
`linear` for those channels.

`storageColorSpace` is per-texture rather than global because a material can legitimately mix an HDR emissive map
that wants `hlg` with a displacement bake that wants `linear`. The `--floatStorage` command line option of
`ntc-cli` sets the same value for all floating point textures that do not specify the field, for manifest-less and
batch use. Use `ntc-cli --describe` to confirm what a file actually ended up with; it reports the storage color
space of every channel.

Note that the loss function is evaluated on stored values: a channel stored as `linear` with a range much wider
than `[0, 1]` will therefore carry more weight in the shared loss than one stored as `hlg`, at the expense of the
other channels in the texture set. Set `lossFunctionScale` appropriately to compensate for this.

## Semantic labels

The semantic labels are case-insensitive, and some of them support multiple versions. The following labels are recognized:

- `Albedo`
- `Alpha`, `Mask`, `AlphaMask`
- `Displ`, `Displacement`
- `Emissive`, `Emission`
- `Glossiness`
- `Metalness`, `Metallic`
- `Normal`
- `Occlusion`, `AO`
- `Roughness`
- `SpecularColor`
- `Transmission`

Note that the semantic labels are currently not stored in the NTC files. Some of them are used by the Explorer app to correctly map the textures to PBR inputs of the material in the 3D view.

The `AlphaMask` (and synonyms) label can be used to enable special processing for the alpha channel. For more information, see the [Settings and Quality Guide](SettingsAndQuality.md).

## Example manifest

```json
{
    "textures": [
        {
            "fileName": "PavingStones070_4K.diffuse.tga",
            "name": "Diffuse",
            "bcFormat": "BC7",
            "isSRGB": true,
            "semantics": {
                "Albedo": "RGB"
            }
        },
        {
            "fileName": "PavingStones070_4K.normal.tga",
            "name": "Normal",
            "bcFormat": "BC7",
            "semantics": {
                "Normal": "RGB"
            }
        },
        {
            "fileName": "PavingStones070_4K.roughness.tga",
            "name": "Roughness",
            "bcFormat": "BC4",
            "semantics": {
                "Roughness": "R"
            }
        }
    ]
}
```