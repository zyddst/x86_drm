#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Crop a large BMP, convert it to a supported pixel format, and tile it into RAW.

This tool follows the bit-field address mappings used by the original
Linear2TileTool.py. RGB/FP16, packed YUY2, and semi-planar YUV use separate paths.
"""
from __future__ import print_function

import argparse
import math
import os
import sys

import cv2
import numpy as np


FORMATS = [
    "ARGB8", "XRGB8", "A2RGB10", "X2RGB10", "RGB565", "ARGB4",
    "A1RGB555", "RGB8", "ABGR2101010", "RGBA1010102",
    "BGRA1010102", "A8RGB565", "L8", "L8A8", "FP16",
    "NV12", "P010", "YUY2", "NV21", "P210", "AYUV",
]

# (tile width, tile height) in the coordinate space of the plane being tiled.
RGB_TILE_SPECS = {
    "linear": (1, 1),
    "16x4": (16, 4),
    "4x4": (4, 4),
    "32x2": (32, 2),
    "32x4": (32, 4),
    "8x8": (8, 8),
    "8x8_unit2x2": (8, 8),
    # The source formula groups 4x4 source pixels into 16 stored pixels.
    "8x4_unit2x2": (4, 4),
    "16x16": (16, 16),
    "32x8": (32, 8),
    "32x8_a": (32, 8),
    "16x8": (16, 8),
    "8x8_supertilex": (64, 64),
}

Y_TILE_SPECS = {
    "linear": (1, 1),
    "32x8": (32, 8),
    "16x16": (16, 16),
    "16x8": (16, 8),
    "8x8": (8, 8),
    # The original 32x8_raster address formula interleaves 8x8 source groups.
    "32x8_raster": (8, 8),
}

UV_TILE_SPECS = {
    "linear": (1, 1),
    "16x8": (16, 8),
    "16x4": (16, 4),
    # The original UV formula uses 2 x bits + 3 y bits for each 32-sample tile.
    "8x8": (4, 8),
    "32x4": (2, 4),
    # The original formula interleaves 3 x bits and 3 y bits (8x8 samples).
    "8x4": (8, 8),
}

RGB_FORMATS = {
    "ARGB8", "XRGB8", "A2RGB10", "X2RGB10", "RGB565", "ARGB4",
    "A1RGB555", "RGB8", "ABGR2101010", "RGBA1010102",
    "BGRA1010102", "A8RGB565", "L8", "L8A8", "FP16",
}
YUV420_FORMATS = {"NV12", "NV21", "P010"}
YUV422_FORMATS = {"YUY2", "P210"}


def get_bits(value, high_bit, low_bit):
    """Extract inclusive bit range [high_bit:low_bit] from an integer."""
    if high_bit < low_bit:
        return 0
    return (value >> low_bit) & ((1 << (high_bit - low_bit + 1)) - 1)


def set_bits(value, high_bit, low_bit, new_val):
    """Replace inclusive bit range [high_bit:low_bit] in an integer."""
    length = high_bit - low_bit + 1
    mask = ((1 << length) - 1) << low_bit
    return (value & ~mask) | ((new_val & ((1 << length) - 1)) << low_bit)


def bits_array(value, high_bit, low_bit):
    """NumPy-array equivalent of get_bits()."""
    if high_bit < low_bit:
        return np.zeros_like(value)
    return (value >> low_bit) & ((1 << (high_bit - low_bit + 1)) - 1)


def align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def parse_size(size_text):
    """Parse WIDTHxHEIGHT (also accepts WIDTH,HEIGHT)."""
    sep = "x" if "x" in size_text.lower() else ","
    parts = size_text.lower().split(sep)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT, e.g. 1920x1080")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        # Python 2.7 does not support exception chaining (raise ... from ...).
        raise argparse.ArgumentTypeError("size must contain integer width and height")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def load_and_crop_bmp(input_path, width, height):
    """Load an 8-bit BMP and crop the top-left WIDTH x HEIGHT region."""
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("cannot read input BMP: {}".format(input_path))
    if image.dtype != np.uint8:
        raise ValueError(
            "only 8-bit BMP input is supported; detected dtype {}".format(image.dtype)
        )
    src_h, src_w = image.shape[:2]
    if src_w < width or src_h < height:
        raise ValueError(
            "requested crop {}x{} exceeds BMP dimensions {}x{}".format(
                width, height, src_w, src_h
            )
        )

    # Crop before format conversion so a large 8K source only converts the requested ROI.
    cropped = image[:height, :width]
    if cropped.ndim == 2:
        bgr = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        alpha = np.full((height, width), 255, dtype=np.uint8)
    elif cropped.ndim == 3 and cropped.shape[2] == 3:
        bgr = cropped
        alpha = np.full((height, width), 255, dtype=np.uint8)
    elif cropped.ndim == 3 and cropped.shape[2] == 4:
        bgr = cropped[:, :, :3]
        # Copy the cropped alpha so it does not keep the full 8K BGRA allocation alive.
        alpha = cropped[:, :, 3].copy()
    else:
        raise ValueError("unsupported BMP channel layout: {}".format(cropped.shape))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, alpha, (src_w, src_h)


def rgb_to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def convert_rgb_bmp(rgb, alpha, fmt, rgb8_order="BGR", fp16_order="BGRA", l8a8_order="AL"):
    """Convert cropped RGB/alpha arrays to the requested packed RGB/gray format.

    Pixel rows are packed incrementally for 16/32-bit formats so a full 8K frame
    does not require several full-frame uint32 channel copies at the same time.
    """
    h, w = rgb.shape[:2]
    r8 = rgb[:, :, 0]
    g8 = rgb[:, :, 1]
    b8 = rgb[:, :, 2]
    a8 = alpha

    if fmt in ("ARGB8", "XRGB8"):
        # Little-endian memory bytes for the AARRGGBB word are B, G, R, A.
        return np.stack([b8, g8, r8, a8], axis=-1).tobytes()

    if fmt == "RGB8":
        # Cmodel packed R8G8B8 is interpreted as a 24-bit word on little-endian
        # memory: B, G, R bytes form the numeric 0xRRGGBB word. Allow RGB as
        # an override for consumers that define RGB8 as byte sequence R,G,B.
        if rgb8_order == "BGR":
            return np.stack([b8, g8, r8], axis=-1).tobytes()
        return np.stack([r8, g8, b8], axis=-1).tobytes()

    if fmt in ("A2RGB10", "X2RGB10", "ABGR2101010", "RGBA1010102", "BGRA1010102"):
        packed = np.empty((h, w), dtype=np.uint32)
        for y in range(h):
            r = r8[y].astype(np.uint32) << 2
            g = g8[y].astype(np.uint32) << 2
            b = b8[y].astype(np.uint32) << 2
            a2 = np.uint32(3)
            if fmt in ("A2RGB10", "X2RGB10"):
                packed[y] = (a2 << 30) | (r << 20) | (g << 10) | b
            elif fmt == "ABGR2101010":
                packed[y] = (a2 << 30) | (b << 20) | (g << 10) | r
            elif fmt == "RGBA1010102":
                packed[y] = (r << 22) | (g << 12) | (b << 2) | np.uint32(3)
            else:  # BGRA1010102
                packed[y] = (b << 22) | (g << 12) | (r << 2) | np.uint32(3)
        return packed.astype("<u4", copy=False).tobytes()

    if fmt in ("RGB565", "A1RGB555", "ARGB4"):
        packed = np.empty((h, w), dtype=np.uint16)
        for y in range(h):
            r = r8[y].astype(np.uint16)
            g = g8[y].astype(np.uint16)
            b = b8[y].astype(np.uint16)
            if fmt == "RGB565":
                packed[y] = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
            elif fmt == "A1RGB555":
                packed[y] = (1 << 15) | ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
            else:
                a = a8[y].astype(np.uint16)
                packed[y] = ((a >> 4) << 12) | ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)
        return packed.astype("<u2", copy=False).tobytes()

    if fmt == "A8RGB565":
        # 24-bit value: A[23:16] | RGB565[15:0], serialized little-endian as 3 bytes/pixel.
        packed = np.empty((h, w), dtype=np.uint32)
        for y in range(h):
            r = r8[y].astype(np.uint32)
            g = g8[y].astype(np.uint32)
            b = b8[y].astype(np.uint32)
            a = a8[y].astype(np.uint32)
            rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
            packed[y] = (a << 16) | rgb565
        bytes4 = packed.astype("<u4", copy=False).view(np.uint8).reshape(h, w, 4)
        return bytes4[:, :, :3].copy().tobytes()

    if fmt == "L8":
        return rgb_to_gray(rgb).tobytes()

    if fmt == "L8A8":
        gray = rgb_to_gray(rgb)
        if l8a8_order == "AL":
            return np.stack([alpha, gray], axis=-1).tobytes()
        return np.stack([gray, alpha], axis=-1).tobytes()

    if fmt == "FP16":
        # Each component is IEEE-754 binary16 in little-endian byte order.
        # The component order is selectable because FP16 container conventions
        # differ between IPs; default BGRA matches the Cmodel's packed-word
        # channel convention. Use --fp16-order RGBA if its reader expects RGBA.
        components = {
            "R": rgb[:, :, 0].astype(np.float32),
            "G": rgb[:, :, 1].astype(np.float32),
            "B": rgb[:, :, 2].astype(np.float32),
            "A": alpha.astype(np.float32),
        }
        order = fp16_order
        fp16 = np.empty((h, w, 4), dtype="<f2")
        for index, component in enumerate(order):
            fp16[:, :, index] = components[component] / 255.0
        return fp16.tobytes()

    raise ValueError("unsupported packed RGB format: {}".format(fmt))


def rgb_to_yuv601_limited(rgb):
    """BT.601 limited-range 8-bit Y, U(full-res), V(full-res) conversion."""
    h, w = rgb.shape[:2]
    y_out = np.empty((h, w), dtype=np.uint8)
    u_out = np.empty((h, w), dtype=np.float32)
    v_out = np.empty((h, w), dtype=np.float32)
    # Process one row at a time to keep temporary float arrays small for 8K inputs.
    for row in range(h):
        r = rgb[row, :, 0].astype(np.float32)
        g = rgb[row, :, 1].astype(np.float32)
        b = rgb[row, :, 2].astype(np.float32)
        y_line = 16.0 + 0.256788 * r + 0.504129 * g + 0.097906 * b
        u_line = 128.0 - 0.148223 * r - 0.290993 * g + 0.439216 * b
        v_line = 128.0 + 0.439216 * r - 0.367788 * g - 0.071427 * b
        y_out[row] = np.clip(np.rint(y_line), 0, 255).astype(np.uint8)
        u_out[row] = np.clip(u_line, 0, 255)
        v_out[row] = np.clip(v_line, 0, 255)
    return y_out, u_out, v_out


def downsample_chroma(u, v, horizontal_subsample, vertical_subsample):
    h, w = u.shape
    if w % horizontal_subsample or h % vertical_subsample:
        raise ValueError("YUV subsampling requires dimensions aligned to {}x{}".format(
            horizontal_subsample, vertical_subsample
        ))
    out_h = h // vertical_subsample
    out_w = w // horizontal_subsample
    u_ds = u.reshape(out_h, vertical_subsample, out_w, horizontal_subsample).mean(axis=(1, 3))
    v_ds = v.reshape(out_h, vertical_subsample, out_w, horizontal_subsample).mean(axis=(1, 3))
    u_ds = np.clip(np.rint(u_ds), 0, 255).astype(np.uint8)
    v_ds = np.clip(np.rint(v_ds), 0, 255).astype(np.uint8)
    return u_ds, v_ds


def convert_bmp_to_yuv(rgb, fmt):
    """Convert cropped RGB into the format's linear plane byte strings."""
    h, w = rgb.shape[:2]
    y, u_full, v_full = rgb_to_yuv601_limited(rgb)

    if fmt in ("NV12", "NV21", "P010"):
        if w % 2 or h % 2:
            raise ValueError("{} requires even crop width and height".format(fmt))
        u, v = downsample_chroma(u_full, v_full, 2, 2)
        if fmt == "NV12":
            uv = np.stack([u, v], axis=-1).astype(np.uint8).tobytes()
            return {"y": y.tobytes(), "uv": uv, "y_bpp": 1, "uv_bpp": 2,
                    "uv_w": w // 2, "uv_h": h // 2}
        if fmt == "NV21":
            vu = np.stack([v, u], axis=-1).astype(np.uint8).tobytes()
            return {"y": y.tobytes(), "uv": vu, "y_bpp": 1, "uv_bpp": 2,
                    "uv_w": w // 2, "uv_h": h // 2}
        # P010 stores 10-bit values in the high 10 bits of each little-endian 16-bit word.
        y10_msb = (y.astype(np.uint16) << 8)  # (8-bit code * 4) << 6
        u10_msb = (u.astype(np.uint16) << 8)
        v10_msb = (v.astype(np.uint16) << 8)
        uv_words = np.stack([u10_msb, v10_msb], axis=-1).astype("<u2")
        return {"y": y10_msb.astype("<u2").tobytes(), "uv": uv_words.tobytes(),
                "y_bpp": 2, "uv_bpp": 4, "uv_w": w // 2, "uv_h": h // 2}

    if fmt == "P210":
        if w % 2:
            raise ValueError("P210 requires an even crop width")
        u, v = downsample_chroma(u_full, v_full, 2, 1)
        y10_msb = (y.astype(np.uint16) << 8)
        uv_words = np.stack([(u.astype(np.uint16) << 8), (v.astype(np.uint16) << 8)], axis=-1).astype("<u2")
        return {"y": y10_msb.astype("<u2").tobytes(), "uv": uv_words.tobytes(),
                "y_bpp": 2, "uv_bpp": 4, "uv_w": w // 2, "uv_h": h}

    if fmt == "YUY2":
        if w % 2:
            raise ValueError("YUY2 requires an even crop width")
        u, v = downsample_chroma(u_full, v_full, 2, 1)
        yuyv = np.empty((h, w // 2, 4), dtype=np.uint8)
        yuyv[:, :, 0] = y[:, 0::2]
        yuyv[:, :, 1] = u
        yuyv[:, :, 2] = y[:, 1::2]
        yuyv[:, :, 3] = v
        # Linear YUY2 memory is Y0,U,Y1,V; the tiler treats each 2-byte pixel pair as one unit.
        return {"pixels": yuyv.reshape(h, w, 2).tobytes(), "bpp": 2}
    
    if fmt == "AYUV":
        if alpha is None:
            alpha = np.full((h, w), 255, dtype=np.uint8)

        u8 = np.clip(
            np.rint(u_full), 0, 255
        ).astype(np.uint8)

        v8 = np.clip(
            np.rint(v_full), 0, 255
        ).astype(np.uint8)

        # Stack in little-endian memory order.
        packed = np.stack(
            [v8, u8, y, alpha.astype(np.uint8)],
            axis=-1
        )

        return {
            "pixels": packed.tobytes(),
            "bpp": 4
        }

    raise ValueError("unsupported YUV format: {}".format(fmt))


def map_rgb_coordinates(x, y, tile_mode):
    """Vectorized equivalent of the original RGB/FP16 tile address formulas."""
    if tile_mode == "linear":
        return x, y
    if tile_mode == "16x4":
        xt = bits_array(x, 3, 0) | (bits_array(y, 1, 0) << 4) | (bits_array(x, 31, 4) << 6)
        yt = y & ~3
    elif tile_mode == "4x4":
        xt = bits_array(x, 1, 0) | (bits_array(y, 1, 0) << 2) | (bits_array(x, 31, 2) << 4)
        yt = y & ~3
    elif tile_mode == "32x2":
        xt = bits_array(x, 4, 0) | (bits_array(y, 0, 0) << 5) | (bits_array(x, 31, 5) << 6)
        yt = y & ~1
    elif tile_mode == "32x4":
        xt = bits_array(x, 4, 0) | (bits_array(y, 1, 0) << 5) | (bits_array(x, 31, 5) << 7)
        yt = y & ~3
    elif tile_mode == "8x8":
        xt = bits_array(x, 2, 0) | (bits_array(y, 2, 0) << 3) | (bits_array(x, 31, 3) << 6)
        yt = y & ~7
    elif tile_mode == "8x8_unit2x2":
        xt = (bits_array(x, 0, 0) | (bits_array(y, 0, 0) << 1) |
              (bits_array(x, 1, 1) << 2) | (bits_array(y, 1, 1) << 3) |
              (bits_array(x, 2, 2) << 4) | (bits_array(y, 2, 2) << 5) |
              (bits_array(x, 31, 3) << 6))
        yt = y & ~7
    elif tile_mode == "8x4_unit2x2":
        xt = (bits_array(x, 0, 0) | (bits_array(y, 0, 0) << 1) |
              (bits_array(x, 1, 1) << 2) | (bits_array(y, 1, 1) << 3) |
              (bits_array(x, 31, 2) << 4))
        yt = y & ~3
    elif tile_mode == "16x16":
        xt = bits_array(x, 3, 0) | (bits_array(y, 3, 0) << 4) | (bits_array(x, 31, 4) << 8)
        yt = bits_array(y, 31, 4) << 4
    elif tile_mode in ("32x8", "32x8_a"):
        xt = bits_array(x, 4, 0) | (bits_array(y, 2, 0) << 5) | (bits_array(x, 31, 5) << 8)
        yt = bits_array(y, 31, 3) << 3
    elif tile_mode == "16x8":
        xt = bits_array(x, 3, 0) | (bits_array(y, 2, 0) << 4) | (bits_array(x, 31, 4) << 7)
        yt = bits_array(y, 31, 3) << 3
    elif tile_mode == "8x8_supertilex":
        xt = ((bits_array(x, 1, 0) << 0) | (bits_array(y, 1, 0) << 2) |
              (bits_array(x, 2, 2) << 4) | (bits_array(y, 2, 2) << 5) |
              (bits_array(x, 3, 3) << 6) | (bits_array(y, 3, 3) << 7) |
              (bits_array(x, 4, 4) << 8) | (bits_array(y, 4, 4) << 9) |
              (bits_array(x, 5, 5) << 10) | (bits_array(y, 5, 5) << 11) |
              (bits_array(x, 31, 6) << 12))
        yt = y & ~63
    else:
        raise ValueError("unsupported RGB/FP16 tile mode: {}".format(tile_mode))
    return xt, yt


def map_y_coordinates(x, y, tile_mode):
    """Vectorized equivalent of the original Y-plane mapping formulas."""
    if tile_mode == "linear":
        return x, y
    if tile_mode == "32x8":
        xt = bits_array(x, 4, 0) | (bits_array(y, 2, 0) << 5) | (bits_array(x, 31, 5) << 8)
        yt = bits_array(y, 31, 3) << 3
    elif tile_mode == "16x16":
        xt = bits_array(x, 3, 0) | (bits_array(y, 3, 0) << 4) | (bits_array(x, 31, 4) << 8)
        yt = bits_array(y, 31, 4) << 4
    elif tile_mode == "16x8":
        xt = bits_array(x, 3, 0) | (bits_array(y, 2, 0) << 4) | (bits_array(x, 31, 4) << 7)
        yt = bits_array(y, 31, 3) << 3
    elif tile_mode == "8x8":
        xt = bits_array(x, 2, 0) | (bits_array(y, 2, 0) << 3) | (bits_array(x, 31, 3) << 6)
        yt = bits_array(y, 31, 3) << 3
    elif tile_mode == "32x8_raster":
        xt = ((bits_array(x, 1, 0) << 0) | (bits_array(y, 1, 0) << 2) |
              (bits_array(x, 2, 2) << 4) | (bits_array(y, 2, 2) << 5) |
              (bits_array(x, 31, 3) << 6))
        yt = y & ~7
    else:
        raise ValueError("unsupported Y-plane tile mode: {}".format(tile_mode))
    return xt, yt


def map_uv_coordinates(x, y, tile_mode):
    """Vectorized equivalent of the original UV-plane mapping formulas."""
    if tile_mode == "linear":
        return x, y
    if tile_mode == "16x8":
        xt = bits_array(x, 3, 0) | (bits_array(y, 2, 0) << 4) | (bits_array(x, 31, 4) << 7)
        yt = bits_array(y, 31, 3) << 3
    elif tile_mode == "16x4":
        xt = bits_array(x, 3, 0) | (bits_array(y, 1, 0) << 4) | (bits_array(x, 31, 4) << 6)
        yt = bits_array(y, 31, 2) << 2
    elif tile_mode == "8x8":
        xt = bits_array(x, 1, 0) | (bits_array(y, 2, 0) << 2) | (bits_array(x, 31, 2) << 5)
        yt = bits_array(y, 31, 3) << 3
    elif tile_mode == "32x4":
        xt = bits_array(x, 0, 0) | (bits_array(y, 1, 0) << 1) | (bits_array(x, 31, 1) << 3)
        yt = y & ~3
    elif tile_mode == "8x4":
        xt = (bits_array(x, 2, 0) | (bits_array(y, 2, 0) << 3) |
              (bits_array(x, 31, 3) << 6))
        yt = bits_array(y, 31, 3) << 3
    else:
        raise ValueError("unsupported UV-plane tile mode: {}".format(tile_mode))
    return xt, yt

def tile_plane(linear_data, width, height, bytes_per_sample, tile_mode, tile_specs, coordinate_mapper, plane_name):
    """Tile a plane and zero-pad its right/bottom edges to the mapping's tile size."""
    if tile_mode not in tile_specs:
        raise ValueError("unsupported {} tile mode '{}'; supported: {}".format(
            plane_name, tile_mode, ", ".join(tile_specs.keys())
        ))
    expected = width * height * bytes_per_sample
    if len(linear_data) != expected:
        raise ValueError("{} plane length mismatch: expected {} bytes, got {}".format(
            plane_name, expected, len(linear_data)
        ))

    tile_w, tile_h = tile_specs[tile_mode]
    padded_w = align_up(width, tile_w)
    padded_h = align_up(height, tile_h)
    src = np.frombuffer(linear_data, dtype=np.uint8).reshape(height, width, bytes_per_sample)
    out = np.zeros((padded_h, padded_w, bytes_per_sample), dtype=np.uint8)

    # Map one source row at a time. This preserves the bit-exact formulas while
    # avoiding full-frame int64 coordinate arrays for 8K images.
    x = np.arange(width, dtype=np.int64)
    out_flat = out.reshape(-1, bytes_per_sample)
    capacity = padded_w * padded_h
    for row in range(height):
        xt, yt = coordinate_mapper(x, np.int64(row), tile_mode)
        dst_index = np.asarray(yt * padded_w + xt, dtype=np.int64).reshape(-1)
        if dst_index.size and (dst_index.min() < 0 or dst_index.max() >= capacity):
            raise ValueError(
                "{} tile mapping produced out-of-range destination index at row {} "
                "(max {}, capacity {})".format(
                    plane_name, row, int(dst_index.max()), capacity
                )
            )
        out_flat[dst_index] = src[row]
    return out.tobytes(), padded_w, padded_h


def tile_packed_image(linear_data, width, height, bpp, tile_mode):
    tiled, padded_w, padded_h = tile_plane(
        linear_data, width, height, bpp, tile_mode,
        RGB_TILE_SPECS, map_rgb_coordinates, "pixel"
    )
    return tiled, padded_w, padded_h


def tile_yuv_image(planes, width, height, fmt, tile_mode):
    if "_uv" not in tile_mode:
        raise ValueError("semi-planar YUV tilemode must have Y and UV parts, e.g. 32x8_uv8x8")
    if tile_mode.count("_uv") != 1:
        raise ValueError("invalid YUV tilemode '{}': expected one '_uv' separator".format(tile_mode))
    y_mode, uv_mode = tile_mode.split("_uv", 1)
    if not y_mode or not uv_mode:
        raise ValueError("invalid YUV tilemode '{}': Y and UV modes are required".format(tile_mode))

    y_tiled, y_pw, y_ph = tile_plane(
        planes["y"], width, height, planes["y_bpp"], y_mode,
        Y_TILE_SPECS, map_y_coordinates, "Y"
    )
    uv_tiled, uv_pw, uv_ph = tile_plane(
        planes["uv"], planes["uv_w"], planes["uv_h"], planes["uv_bpp"], uv_mode,
        UV_TILE_SPECS, map_uv_coordinates, "UV"
    )
    return y_tiled + uv_tiled, (y_pw, y_ph), (uv_pw, uv_ph)


def ensure_output_suffix(path, output_format):
    """Normalize the output extension for the selected container format."""
    wanted = ".bmp" if output_format == "bmp" else ".raw"
    root, ext = os.path.splitext(path)
    if ext.lower() == wanted:
        return path
    # Replace a mismatching extension instead of producing names like .raw.bmp.
    return (root if ext else path) + wanted


def linear_rgb_to_bmp_bytes(linear, fmt, width, height, rgb8_order="BGR"):
    """Decode supported Linear RGB RAW bytes into OpenCV-ready BGR pixels."""
    if fmt == "RGB8":
        pixels = np.frombuffer(linear, dtype=np.uint8).reshape((height, width, 3))
        if rgb8_order == "BGR":
            return pixels.copy()
        return pixels[:, :, ::-1].copy()

    if fmt == "ARGB8":
        # convert_rgb_bmp stores little-endian AARRGGBB words as B,G,R,A bytes.
        pixels = np.frombuffer(linear, dtype=np.uint8).reshape((height, width, 4))
        return pixels[:, :, :3].copy()

    if fmt == "A2RGB10":
        words = np.frombuffer(linear, dtype="<u4").reshape((height, width))
        r10 = (words >> 20) & 0x3ff
        g10 = (words >> 10) & 0x3ff
        b10 = words & 0x3ff
        # BMP is 8-bit/channel; convert the 10-bit values back to 8-bit for viewing.
        r8 = np.minimum(255, r10 >> 2).astype(np.uint8)
        g8 = np.minimum(255, g10 >> 2).astype(np.uint8)
        b8 = np.minimum(255, b10 >> 2).astype(np.uint8)
        return np.stack([b8, g8, r8], axis=-1)

    raise ValueError("BMP output is only supported for RGB8, ARGB8 and A2RGB10")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Crop a BMP to the requested size, convert its format, and output Linear or tiled RAW."
    )
    parser.add_argument("--input", required=True, help="Input 8-bit BMP file (e.g. source_8k.bmp)")
    parser.add_argument("--size", required=True, type=parse_size,
                        help="Output crop size WIDTHxHEIGHT, e.g. 1920x1080")
    parser.add_argument("--format", "--fmt", dest="fmt", required=True, choices=FORMATS,
                        help="Output pixel format")
    parser.add_argument("--tilemode", required=True,
                        help="Tile mode; semi-planar YUV uses YMODE_uvUVMODE")
    parser.add_argument("--output", help="Output path; extension is normalized to match --output-format")
    parser.add_argument("--output-format", choices=["raw", "bmp"], default="raw",
                        help="Output container: raw (default) or BMP; BMP is supported only for RGB8/ARGB8/A2RGB10 with --tilemode linear")
    parser.add_argument("--rgb8-order", choices=["BGR", "RGB"], default="BGR",
                        help="RGB8 byte order (default BGR for little-endian R8G8B8 word layout)")
    parser.add_argument("--fp16-order", choices=["BGRA", "RGBA"], default="BGRA",
                        help="FP16 component order (default BGRA; use RGBA if required by the Cmodel format)")
    parser.add_argument("--l8a8-order", choices=["LA", "AL"], default="AL",
                        help="L8A8 byte order: alpha-luminance (default AL, matching Cmodel interpretation observed in testing)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    width, height = args.size
    fmt = args.fmt

    try:
        if args.output_format == "bmp":
            if fmt not in ("RGB8", "ARGB8", "A2RGB10"):
                raise ValueError("--output-format bmp is supported only for RGB8, ARGB8 and A2RGB10")
            if args.tilemode != "linear":
                raise ValueError("--output-format bmp requires --tilemode linear")
        rgb, alpha, source_size = load_and_crop_bmp(args.input, width, height)
        print("Input BMP: {} ({}x{})".format(args.input, source_size[0], source_size[1]))
        print("Crop region: top-left {}x{}".format(width, height))
        print("Output format: {}".format(fmt))
        print("Tile mode: {}".format(args.tilemode))
        if fmt == "RGB8":
            print("RGB8 byte order: {}".format(args.rgb8_order))
        elif fmt == "FP16":
            print("FP16 component order: {}".format(args.fp16_order))
        elif fmt == "L8A8":
            print("L8A8 byte order: {}".format(args.l8a8_order))

        if fmt in RGB_FORMATS:
            linear = convert_rgb_bmp(
                rgb, alpha, fmt, args.rgb8_order, args.fp16_order, args.l8a8_order
            )
            bpp = {
                "ARGB8": 4, "XRGB8": 4, "A2RGB10": 4, "X2RGB10": 4,
                "ABGR2101010": 4, "RGBA1010102": 4, "BGRA1010102": 4,
                "RGB565": 2, "ARGB4": 2, "A1RGB555": 2,
                "RGB8": 3, "A8RGB565": 3, "L8": 1, "L8A8": 2, "FP16": 8,
            }[fmt]
            if args.tilemode == "linear":
                # Linear mode writes the converted pixels exactly in row-major order.
                # No Tile coordinate mapping or edge padding is performed.
                tiled = linear
                print("Linear RAW: {}x{} pixels; no Tile mapping or edge padding".format(width, height))
            else:
                tiled, padded_w, padded_h = tile_packed_image(
                    linear, width, height, bpp, args.tilemode
                )
                print("Tiled storage size: {}x{} pixels (edge padding may be present)".format(padded_w, padded_h))

        elif fmt == "AYUV":
            if args.tilemode != "linear":
                raise ValueError(
                    "AYUV currently supports --tilemode linear only"
                )

            planes = convert_bmp_to_yuv(
                rgb, fmt, alpha
            )

            tiled = planes["pixels"]

            print(
                "AYUV byte order: V, U, Y, A "
                "(little-endian A:Y:U:V word)"
            )

            print(
                "Linear RAW: {}x{} pixels; "
                "4 bytes/pixel; no Tile mapping or edge padding"
                .format(width, height)
            )
        elif fmt == "YUY2":
            if args.tilemode not in ("linear", "4x4"):
                raise ValueError("YUY2 supports only tile modes: linear, 4x4")
            planes = convert_bmp_to_yuv(rgb, fmt)
            if args.tilemode == "linear":
                tiled = planes["pixels"]
                print("Linear RAW: {}x{} pixels; no Tile mapping or edge padding".format(width, height))
            else:
                tiled, padded_w, padded_h = tile_packed_image(
                    planes["pixels"], width, height, planes["bpp"], args.tilemode
                )
                print("Tiled storage size: {}x{} pixels (edge padding may be present)".format(padded_w, padded_h))
        else:
            planes = convert_bmp_to_yuv(rgb, fmt)
            if args.tilemode == "linear":
                # For semi-planar YUV, the generic 'linear' option means linear
                # Y plane followed immediately by the linear UV/VU plane.
                tiled = planes["y"] + planes["uv"]
                y_bytes = len(planes["y"])
                uv_bytes = len(planes["uv"])
                print("Linear Y plane: {} bytes; UV plane: {} bytes".format(y_bytes, uv_bytes))
                print("Linear RAW: no Tile mapping or edge padding")
            else:
                tiled, y_dims, uv_dims = tile_yuv_image(planes, width, height, fmt, args.tilemode)
                print("Tiled Y storage: {}x{} samples; UV storage: {}x{} samples".format(
                    y_dims[0], y_dims[1], uv_dims[0], uv_dims[1]
                ))
        default_ext = "bmp" if args.output_format == "bmp" else "raw"
        output_path = args.output or "{}_{}x{}_{}_{}.{}".format(
            os.path.splitext(args.input)[0], width, height, fmt, args.tilemode, default_ext
        )
        output_path = ensure_output_suffix(output_path, args.output_format)
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if args.output_format == "bmp":
            bgr_image = linear_rgb_to_bmp_bytes(tiled, fmt, width, height, args.rgb8_order)
            if not cv2.imwrite(output_path, bgr_image):
                raise IOError("failed to write BMP output: {}".format(output_path))
            output_size = os.path.getsize(output_path)
            print("Output BMP: {}".format(output_path))
        else:
            with open(output_path, "wb") as output_file:
                output_file.write(tiled)
            output_size = len(tiled)
            print("Output RAW: {}".format(output_path))

        print("Output size: {} bytes".format(output_size))
        print("Conversion completed successfully.")
        return 0
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
