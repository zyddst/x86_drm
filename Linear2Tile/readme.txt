# Linear2Tile Tool

## 1. Overview

`Linear2Tile` is a command-line utility that:

1. Loads an 8-bit BMP image and **crops the top-left `WIDTH x HEIGHT` region**;
2. **Converts the cropped pixels** into one of the supported pixel formats  
   (packed RGB / gray / FP16, packed YUY2, semi-planar YUV, AYUV);
3. **Tiles the linear pixel data** into the requested tile mode (bit-field  
   address mapping), or keeps it linear when `--tilemode linear` is given;
4. Writes the result as a **RAW** binary (or a viewable **BMP** for a few  
   linear RGB formats, which is handy for visual verification).

The tool follows the same bit-field address formulas as the original  
`Linear2TileTool.py`. RGB/FP16, packed YUY2, and semi-planar YUV planes use  
separate mapping paths. Images are processed row by row internally, so large  
sources (e.g. 8K BMPs) can be cropped and converted without huge temporary  
allocations.

## 2. Requirements

| Item         | Value                                    |
| ------------ | ---------------------------------------- |
| Python       | 2.7.15 (also compatible with Python 3.x) |
| OpenCV (cv2) | 4.2 (any recent 4.x works)               |
| NumPy        | 1.15+                                    |

On hosts without Internet access, install the dependencies offline, e.g.:

```bash
pip install --no-index --find-links <local_wheels_dir> opencv-python numpy
```

## 3. Supported Formats and Tile Modes

The table below lists the **validated format / tile-mode combinations** and  
the corresponding entries in the Cmodel self-test json.

> `linear` is always accepted as a tile mode for every format (linear output,  
> no tile mapping and no edge padding). It is omitted from the table for  
> brevity.

| Format   | Tile mode                                                                 | Cmodel json format | Cmodel json tile mode                                                         |
| -------- | ---------------------------------------------------------------------     | ------------------ | ------------------------------------------------------------------------      |
| ARGB8    | 16x16 4x4 8x8_supertilex 8x8_unit2x2 32x8_a 16x4                          | AR24               | TILE_16X16 TILE_4X4 SUPER_TILED_XMAJOR TILE_8X8_UNIT2X2 TILE_32X8_A TILE_16X4 |
| A8RGB565 | 4x4                                                                       | AR16               | TILE_4X4                                                                      |
| A1RGB555 | 4x4 32x4                                                                  | AR15               | TILE_4X4 TILE_32X4                                                            |
| A2RGB10  | 4x4 8x8_supertilex 8x8_unit2x2 16x16 32x8 16x4                            | AR30               | TILE_4X4 SUPER_TILED_XMAJOR TILE_8X8_UNIT2X2 TILE_16X16 TILE_32X8 TILE_16X4   |
| ARGB4    | 4x4 32x4                                                                  | AR12               | TILE_4X4 TILE_32X4                                                            |
| RGB8     | 4x4                                                                       | RG24               | TILE_4X4                                                                      |
| RGB565   | 4x4 32x4                                                                  | RG16               | TILE_4X4 TILE_32X4                                                            |
| L8A8     | 4x4                                                                       | A8L8               | TILE_4X4                                                                      |
| L8       | 4x4                                                                       | LUMA_8             | TILE_4X4                                                                      |
| P210     | 8x8_uv8x8                                                                 | P210_UV            | TILE_8X8                                                                      |
| P010     | 16x16_uv8x8 16x8_uv8x8 32x8_raster_uv32x4 32x8_uv16x4                     | P010_UV            | TILE_16X16 TILE_16X8 TILE_16X8_YUYVSP8X8 TILE_32X8_A                          |
| NV12     | 32x8_raster_uv32x4 32x8_uv16x4 32x8_uv16x8 32x8_raster_uv32x4 16x16_uv8x8 | NV12               | TILE_32X8_YUYVSP8X8 TILE_32X8_A TILE_32X8 TILE_16X8_YUYVSP8X8 TILE_16X16      |
| YUY2     | 4x4                                                                       | YUYV               | TILE_4X4                                                                      |
| FP16     | 8x4_unit2x2 32x2                                                          | AR4H               | TILE_8X4_UNIT2X2 TILE_32X2                                                    |

**Unsupported formats** (not supported by this tool now):  
`I010`, `NV16`, `YV12`, `RGB10`.

Additional notes:

- Semi-planar YUV tile modes combine a Y-plane mode and a UV-plane mode,  
  written as `YMODE_uvUVMODE`, e.g. `32x8_raster_uv32x4`.
- `YUY2` supports `linear` and `4x4` only; `AYUV` supports `linear` only.
- `NV12`/`P010` require even crop width and height; `P210`/`YUY2` require an  
  even crop width.
- When a tile mode is selected, the right/bottom edges are zero-padded up to  
  the tile size; the actual tiled storage size is printed at run time.
- For `RGB8`, `FP16`, and `L8A8`, the byte/component order is selectable via  
  command-line options (see Section 4) to match the Cmodel convention  
  (defaults: `BGR`, `BGRA`, and `AL` respectively).

## 4. Command-Line Parameters

```
python Linear2TileTool.py --input <file.bmp> --size <WxH> --format <FORMAT>
                      --tilemode <MODE> [--output <path>]
                      [--output-format raw|bmp]
                      [--rgb8-order BGR|RGB] [--fp16-order BGRA|RGBA]
                      [--l8a8-order AL|LA]
```

| Parameter            | Required | Description                                                                                                                                                                                        |
| -------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--input`            | Yes      | Input 8-bit BMP file (e.g. `source_8k.bmp`). May be larger than the crop size; only the top-left `W x H` region is used.                                                                           |
| `--size`             | Yes      | Output crop size, `WIDTHxHEIGHT` (also accepts `WIDTH,HEIGHT`), e.g. `1920x1080`.                                                                                                                  |
| `--format`		   | Yes      | Output pixel format. One of: `ARGB8, XRGB8, A2RGB10, X2RGB10, RGB565, ARGB4, A1RGB555, RGB8, ABGR2101010, RGBA1010102, BGRA1010102, A8RGB565, L8, L8A8, FP16, NV12, P010, YUY2, NV21, P210, AYUV`. |
| `--tilemode`         | Yes      | Tile mode. Use `linear` for no tiling; semi-planar YUV formats use the `YMODE_uvUVMODE` syntax (e.g. `32x8_uv8x8`). See Section 3 for validated combinations.                                      |
| `--output`           | No       | Output file path. Default: `<input_root>_<W>x<H>_<FORMAT>_<tilemode>.raw` next to the input. The extension is normalized automatically (`.raw` or `.bmp`).                                         |
| `--output-format`    | No       | `raw` (default) or `bmp`. BMP output is only supported for `RGB8` / `ARGB8` / `A2RGB10` with `--tilemode linear`, and is intended for visual verification.                                         |
| `--rgb8-order`       | No       | `RGB8` byte order: `BGR` (default, little-endian R8G8B8 word layout) or `RGB` (byte sequence R,G,B).                                                                                               |
| `--fp16-order`       | No       | `FP16` component order: `BGRA` (default, matches the Cmodel packed-word convention) or `RGBA`. Each component is IEEE-754 binary16, little-endian.                                                 |
| `--l8a8-order`       | No       | `L8A8` byte order: `AL` (alpha-luminance, default) or `LA` (luminance-alpha).                                                                                                                      |

## 5. Usage Examples

Packed RGB, tiled:

```bash
python Linear2TileTool.py --input source_8k.bmp --size 1920x1080 --format ARGB8 --tilemode 32x8_a
```

Semi-planar YUV, tiled (Y mode + UV mode):

```bash
python Linear2TileTool.py --input source_8k.bmp --size 1920x1080 -format NV12 --tilemode 32x8_raster_uv32x4
```

Linear output (no tiling, no padding):

```bash
python Linear2TileTool.py --input source_8k.bmp --size 1920x1080 --format P010 --tilemode linear
```

BMP output for visual verification:

```bash
python Linear2TileTool.py --input source_8k.bmp --size 640x480 --format ARGB8 --tilemode linear --output-format bmp --output check.bmp
```

A successful run prints the input size, crop region, format, tile mode,  
tiled storage dimensions (when tiling), and the final output size in bytes.

## 6. Output Conventions

- **RGB formats** ¡ª pixel words are written in little-endian byte order;  
  e.g. `ARGB8` memory bytes are `B, G, R, A`, matching the little-endian  
  `AARRGGBB` word layout.
- **10-bit formats** (`A2RGB10` and friends) ¡ª one little-endian 32-bit word  
  per pixel. `P010`/`P210` store 10-bit samples in the high 10 bits of each  
  little-endian 16-bit word.
- **YUV conversion** ¡ª BT.601 limited range; chroma is averaged over the  
  2x2 (`NV12`/`P010`) or 2x1 (`P210`/`YUY2`) blocks and rounded to 8-bit.  
  `NV12` = `Y` plane + interleaved `UV`; `NV21` = `Y` plane + interleaved `VU`.
- **AYUV** ¡ª 4 bytes/pixel in memory order `V, U, Y, A` (little-endian  
  `A:Y:U:V` word), alpha forced to 255.
- **Tiled output** ¡ª tiles are stored at the addresses given by the tile  
  address mapping; unmapped padding bytes at the right/bottom edges are zero.
