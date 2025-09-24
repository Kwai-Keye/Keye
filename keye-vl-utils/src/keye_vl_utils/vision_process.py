from __future__ import annotations

from typing import Tuple, Dict, Any, Union
import base64
import logging
import math
import time
import warnings
import itertools
import io as py_io
import os.path as osp
import cv2
import random
import numpy as np
import copy
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections

from io import BytesIO
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from einops import rearrange
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
# min tokens per image
MIN_TOKENS = 4
# max tokens per image
MAX_TOKENS = 20480
MIN_PIXELS = MIN_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 4 * 28 * 28 = 3,136
MAX_PIXELS = MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 20480 * 28 * 28 = 16,056,320
MAX_RATIO = 200

# min tokens per video frame
VIDEO_MIN_TOKENS = 48
# max tokens per video frame
VIDEO_MAX_TOKENS = 768
# min pixels per video frame
VIDEO_MIN_PIXELS = VIDEO_MIN_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 48 * 28 * 28 = 37,632
# max pixels per video frame
VIDEO_MAX_PIXELS = VIDEO_MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR # 768 * 28 * 28 = 602,112
# max total pixels per video
VIDEO_TOTAL_PIXELS = 65536 * IMAGE_FACTOR * IMAGE_FACTOR # 65,536 * 28 * 28 = 51,380,224
# default fps
FPS = 2.0

FAST_TOKEN_RATIO = 0.3

# Slow-Fast帧最小相似度，低于阈值需要重新建立Slow帧，降低该阈值会创建更多的Fast帧
MIN_FRAME_SIMILARITY = 0.9

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
        height: int, width: int,
        factor: int = IMAGE_FACTOR,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return max(h_bar, factor), max(w_bar, factor)


def fetch_image(ele: Dict[str, str | Image.Image],
                size_factor: int = IMAGE_FACTOR,
                is_video = False) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")  ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        # 以image list形式传入的视频
        if is_video:
            min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
            max_pixels = ele.get("max_pixels", VIDEO_MAX_PIXELS)
        else:
            min_pixels = ele.get("min_pixels", MIN_PIXELS)
            max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
        ele: dict,
        total_frames: int,
        video_fps: int | float) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    # TODO: 兼容image list形式
    fps = ele.get("fps", FPS) # 应该是走的默认FPS，按照每秒抽两帧来算
    fps = min(fps, video_fps) # 注意，这里的video_fps是真实的后验FPS
    # 计算每帧使用最少token的情况下，能吃多少帧，这个是用来兜底的，最终一个视频的帧数不会超过这个
    max_frames = int(ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS) / \
        ele.get("min_pixels", VIDEO_MIN_PIXELS))
    # 如果用户指定了max_frames，在不超过兜底max_frames的情况下，会优先使用
    max_frames = min(ele.get("max_frames", max_frames), max_frames)
    fps_nframes = int(total_frames / video_fps * fps) # 换算为秒数，之后计算希望抽多少帧
    nframes = min(fps_nframes, max_frames)
    return nframes


def get_frame_sim(frame1, frame2,
                  patch_size: int=28,
                  threshold: float = 0.7,
                  epsilon: float=1e-8):
    assert frame1.dim() == 3 and frame2.dim() == 3, "输入必须是3D张量 [C, H, W]"
    
    # 将PyTorch张量转换为OpenCV格式的numpy数组
    def to_numpy_cvt(tensor):
        # 确保张量在CPU上并转换为HWC格式
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        if tensor.dtype == np.float32 or tensor.dtype == np.float64:
            tensor = (tensor).astype(np.uint8)
        # 转换为HSV颜色空间
        return cv2.cvtColor(tensor, cv2.COLOR_RGB2HSV)

    # 转换颜色空间
    frame1_hsv = to_numpy_cvt(frame1)
    frame2_hsv = to_numpy_cvt(frame2)

    # 将HSV图像转回PyTorch张量
    frame1_tensor = torch.from_numpy(frame1_hsv).permute(2, 0, 1).to(frame1.device).float()
    frame2_tensor = torch.from_numpy(frame2_hsv).permute(2, 0, 1).to(frame2.device).float()

    # 分块处理
    patch1 = rearrange(
        frame1_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()
    patch2 = rearrange(
        frame2_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()

    norm1 = torch.norm(patch1, p=2, dim=-1, keepdim=True) + epsilon
    norm2 = torch.norm(patch2, p=2, dim=-1, keepdim=True) + epsilon
    
    normalized1 = patch1 / norm1
    normalized2 = patch2 / norm2
    cos_sim = (normalized1 * normalized2).sum(dim=-1)
    
    zero_vector_mask = (norm1.squeeze() < 0.01) & (norm2.squeeze() < 0.01)  # 全黑图
    
    similar = torch.ones_like(cos_sim)  # 默认全部相似
    
    non_zero_mask = ~zero_vector_mask
    similar[non_zero_mask] = (cos_sim[non_zero_mask] > threshold).float()

    return similar[non_zero_mask].float().mean().item()

def extract_slow_fast_frames(frames, threshold = MIN_FRAME_SIMILARITY):
    def _extract_slow_indices(frames):
        assert frames.dim() == 4, "输入必须是4D张量 [N, C, H, W]"

        # 首帧一定是Slow
        slow_indices = [0]
        last_key_frame = frames[0]
        for i in range(1, frames.size(0)):
            current_frame = frames[i]
            sim = get_frame_sim(last_key_frame, current_frame)

            if sim < threshold:
                slow_indices.append(i)
                last_key_frame = current_frame  # 更新关键帧
        
        return slow_indices

    _, _, height, width = frames.shape
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=256 * IMAGE_FACTOR * IMAGE_FACTOR,
    )

    resized_frames = nn.functional.interpolate(
        frames,
        [resized_height, resized_width],
        mode="bilinear",
        antialias=True,
    ).float()

    slow_indices = _extract_slow_indices(resized_frames)
    frame_types = torch.ones(size=(frames.size(0), ), dtype=torch.int32)
    frame_types[slow_indices] = 0

    return frame_types

def _read_video_decord(
        ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    st = time.time()
    if isinstance(ele["video"], bytes):
        video_path = ""
        fp = py_io.BytesIO(ele["video"])
        vr = decord.VideoReader(fp)
    else:
        video_path = ele["video"]
        vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    nframes, video_fps = len(vr), vr.get_avg_fps()
    # timestamp start from 0.0
    timestamps = torch.FloatTensor([(1 / video_fps) * i for i in range(nframes)])

    final_nframes = smart_nframes(ele, total_frames=nframes, video_fps=video_fps)

    indices = torch.linspace(0, nframes - 1, final_nframes).round().long()
    frames = vr.get_batch(indices.tolist()).asnumpy()
    frames = torch.tensor(frames).permute(0, 3, 1, 2)
    logger.debug(f"Decord: {video_path=}, {nframes=}, {video_fps=}, time={time.time() - st:.3f}s")
    timestamps = timestamps[indices]

    ##### extract key frames start ######
    threshold = ele.get("min_frame_similarity", MIN_FRAME_SIMILARITY)
    frame_types = extract_slow_fast_frames(frames, threshold)
    ##### extract key frames end ######
    logger.debug(f"Read video:  {video_path=}, {nframes=}, {video_fps=}, time={time.time() - st:.3f}s")

    return frames, timestamps, frame_types


def fetch_video(ele: Dict, image_factor: int = IMAGE_FACTOR) -> Dict[str, Any]:
    if isinstance(ele["video"], str) or isinstance(ele["video"], bytes):
        frames, timestamps, frame_types = _read_video_decord(ele)
    else:
        # TODO: image list没有走smart_nframes，所以可能会超过video_total_pixels
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = []
        for video_element in ele["video"]:
            # preprocess images
            if isinstance(video_element, dict):
                images.append(
                    fetch_image(video_element, size_factor=image_factor, is_video = True))
            else:
                images.append(
                    fetch_image(
                        {"image": video_element, **process_info},
                        size_factor=image_factor, is_video = True)
                )

        images_tensor = [torch.from_numpy(np.array(image)).permute(2, 0, 1) for image in images]
        frames = torch.stack(images_tensor, dim=0)
        nframes = len(images)
        video_fps = ele.get("fps", None)
        timestamps = None
        # 如果用户主动提供了fps，按照fps来估算timestames
        # 如果没有提供，不会按默认的fps去算
        if video_fps:
            assert isinstance(video_fps, Union[float, int]) and video_fps > 0, \
                "Invalid fps, should be float or int"
            timestamps = torch.FloatTensor([(1 / video_fps) * i for i in range(nframes)])
        final_nframes = smart_nframes(ele, total_frames=nframes, video_fps=ele.get("fps", FPS))
        indices = torch.linspace(0, nframes - 1, final_nframes).round().long()
        frames = frames[indices]
        if timestamps is not None:
            timestamps = timestamps[indices]
        threshold = ele.get("min_frame_similarity", MIN_FRAME_SIMILARITY)
        frame_types = extract_slow_fast_frames(frames, threshold)

    ### 计算slow fast的token量 begin ###
    nframes = len(frame_types)
    fast_nframes = int(sum(frame_types))
    slow_nframes = nframes - fast_nframes

    min_pixels = max(int(ele.get("min_pixels", VIDEO_MIN_PIXELS)), VIDEO_MIN_PIXELS)
    min_tokens = int(min_pixels / IMAGE_FACTOR / IMAGE_FACTOR)
    left = min_pixels / IMAGE_FACTOR / IMAGE_FACTOR
    right = ele.get("max_pixels", VIDEO_MAX_PIXELS) / IMAGE_FACTOR / IMAGE_FACTOR
    def _estimate_total_pixels(tokens_per_frame):
        return slow_nframes * tokens_per_frame * IMAGE_FACTOR * IMAGE_FACTOR + \
            fast_nframes * max(int(FAST_TOKEN_RATIO * tokens_per_frame), min_tokens) * \
                IMAGE_FACTOR * IMAGE_FACTOR

    while left < right:
        mid = int(left + right) // 2
        if _estimate_total_pixels(mid) > ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS):
            right = mid
        else:
            left = mid + 1
    slow_max_pixels = left * IMAGE_FACTOR * IMAGE_FACTOR
    # 计算slow fast的token量 end ###

    _, _, height, width = frames.shape

    #### slow part ######
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=slow_max_pixels,
    )
    slow_num_tokens = resized_height * resized_width / IMAGE_FACTOR / IMAGE_FACTOR
    fast_max_pixels = max(
        int(slow_num_tokens * FAST_TOKEN_RATIO) * IMAGE_FACTOR * IMAGE_FACTOR,
        VIDEO_MIN_PIXELS
    )
    # 注意：fast的实际token其实不严格受到min_pixels约束，只会保证最终的一定不会超过fast_max_pixels
    fast_resized_height, fast_resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=fast_max_pixels,
    )
    fast_num_tokens = fast_resized_height * fast_resized_width / IMAGE_FACTOR / IMAGE_FACTOR
    logger.debug(
        f"fetch_video: {nframes=}, {slow_nframes=}, {fast_nframes=}, {slow_num_tokens=}, "
        f"{fast_num_tokens=}, {min_pixels=}, {resized_height=}, {resized_width=}, "
        f"{fast_resized_height=}, {fast_resized_width}"
    )
    processor_kwargs = {
        "height": resized_height,
        "width": resized_width,
        "fast_height": fast_resized_height,
        "fast_width": fast_resized_width,
    }
    if timestamps is not None:
        processor_kwargs["timestamps"] = timestamps
    if frame_types is not None:
        processor_kwargs["frame_types"] = frame_types
    return frames, processor_kwargs

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos

def process_vision_info(
        conversations: list[dict] | list[list[dict]] = None, vision_infos: list[dict] = None,
        image_factor: int = IMAGE_FACTOR
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    assert conversations is not None or vision_infos is not None

    if vision_infos is None:
        vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    processor_kwargs = collections.defaultdict(list)
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_factor))
        elif "video" in vision_info:
            _video_inputs, _processor_kwargs = fetch_video(vision_info, image_factor)
            video_inputs.append(_video_inputs)
            for k, v in _processor_kwargs.items():
                processor_kwargs[k].append(v)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs, processor_kwargs
