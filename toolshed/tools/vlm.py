# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
VLM (Vision-Language Model) tool backed by Molmo via Transformers.

This tool provides two main capabilities:
1. detect_all(image, obj_name):     Locate all instances of `obj_name` in an image and return bounding
   boxes as a list of [x1, y1, x2, y2].
2. general_query(image, query):     Ask an arbitrary question about the image and return the textual
   answer.

The image argument can be either a PIL.Image.Image object _or_ a numpy array.

NOTE: This implementation expects the `transformers` package and Pillow
installed in the environment where the tool actor is launched.  The
actor must request at least one GPU (`resources={"num_gpus": 1}`) when
configured via toolshed's `tool_configs`.

For 8-bit quantization support, install bitsandbytes:
  pip install bitsandbytes
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Tuple

from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult
from PIL import Image, ImageDraw

# Heavy dependencies - imported directly since tools are only instantiated
# inside Ray actors running in tool-specific conda environments where
# dependencies are guaranteed to be available.
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)
import torch  # type: ignore

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]
Point = Tuple[float, float]


class VLMTool(BaseTool):
    """Vision-Language model tool using Molmo via Transformers."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self,
                 model_name: str = "allenai/Molmo-7B-D-0924",
                 dtype: str = "float16",
                 no_output_image: bool = False,
                 no_output_vars: bool = False,
                 exclude_methods: list[str] = None,
                 exclude_behavior: str = "warn",
                 use_8bit_quantization: bool = False) -> None:
        super().__init__(
            no_output_image=no_output_image, 
            no_output_vars=no_output_vars,
            exclude_methods=exclude_methods,
            exclude_behavior=exclude_behavior
        )

        # Create the model – expensive operation done once per actor.
        logger.info("Loading VLM model %s via Transformers …", model_name)

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

        # Configure quantization if requested
        quantization_config = None
        if use_8bit_quantization:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16 if dtype == "float16" else torch.float32
                )
                logger.info("Using 8-bit quantization for faster inference")
            except ImportError:
                logger.warning(
                    "8-bit quantization requested but bitsandbytes not available. "
                    "Install with: pip install bitsandbytes\n"
                    "Falling back to regular loading."
                )
                quantization_config = None

        # Load model with or without quantization
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            # Only set torch_dtype if not using quantization
            model_kwargs["torch_dtype"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        return "vlm"

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------
    
    def _detection_result(self, img: Image.Image, points: list, obj_name: str) -> ToolResult[list]:
        points_str = str(points)
        num_detected = len(points)
        
        # Create overlay visualization if points were detected and not suppressed
        overlay_image = None
        if points and not self.no_output_image:
            # Convert normalized points back to pixel coordinates for visualization
            w, h = img.size
            pixel_points = [(int(x * w), int(y * h)) for x, y in points]
            overlay_image = self._create_point_overlay(img, pixel_points, obj_name)
        
        safe_var_name = f"{obj_name.replace(' ', '_')}_detections"
        
        # Prepare variables only if not suppressed
        variables = {} if self.no_output_vars else ({safe_var_name: points} if points else {})
        
        # Adjust text based on whether variables are suppressed
        text = f"Detected {num_detected} instance(s) of '{obj_name}': {points_str}"
        if points and not self.no_output_vars:
            text += f" Use ${safe_var_name} to reference them."
        
        return ToolResult(
            points,  # The actual list of points for code execution
            text=text,
            image=overlay_image,  # Will be None if no_output_image is True
            variables=variables
        )

    @tool_method
    def detect_one(self, image, obj_name: str) -> ToolResult[list]:
        """Detect *one* instance of *obj_name* in *image*.

        [[if:text]]Text output: coordinates of a single point for the first instance of the object, in normalized pixel space in range [0, 1].[[/if:text]]
        [[if:image]]Visual output: an overlay image with detected object marked as red point.[[/if:image]]
        [[if:vars]]Stored variables: Coordinates in ${obj_name}_detections variable of list of (x, y) point coordinates in normalized pixel space [0, 1] for use in subsequent operations.[[/if:vars]]
        """
        print(f"detect_one called with image: {image} and obj_name: {obj_name}")
        
        img = self._resolve_image(image)
        response_text = self._run_llm(img, f"Point to the {obj_name} in this image.")
        points = self._parse_points(response_text)

        if points:  # check if points list is not empty
            points = [points[0]] # only return the first point
        
        return self._detection_result(img, points, obj_name)

    @tool_method
    def detect_all(self, image, obj_name: str) -> ToolResult[list]:
        """Detect *all* instances of *obj_name* in *image*.

        [[if:text]]Text output: List of point coordinates for the detected objects, in normalized pixel space in range [0, 1].[[/if:text]]
        [[if:image]]Visual output: an overlay image with detected objects marked as red points.[[/if:image]]
        [[if:vars]]Stored variables: Coordinates in ${obj_name}_detections variable of list of (x, y) point coordinates in normalized pixel space [0, 1] for use in subsequent operations.[[/if:vars]]

        Args:
            image (Image): image in which to detect objects.
            obj_name (str): Name or description of the object to detect.

        Returns:
            list: List of (x, y) point coordinates in normalized pixel space [0, 1].
        """
        print(f"detect_all called with image: {image} and obj_name: {obj_name}")
        
        img = self._resolve_image(image)
        response_text = self._run_llm(img, f"Point to all the {obj_name}s in this image.")
        points = self._parse_points(response_text)

        return self._detection_result(img, points, obj_name)

    #@tool_method
    def general_query(self, image, query: str) -> ToolResult[str]:
        """Free-form question answering on *image*.

        [[if:text]]Text output: The answer to the natural-language question about the image.[[/if:text]]

        Args:
            image: PIL image.
            query: Natural-language question.

        Returns:
            str: Answer to the query about the image.
        """
        img = self._resolve_image(image)
        prompt = query
        answer = self._run_llm(img, prompt)
        
        answer_text = answer.strip()
        return ToolResult(
            answer_text,  # The actual value for code execution
            text=answer_text  # Same text for the LLM to see
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_point_overlay(
        image: Image.Image,
        points: List[Tuple[int, int]],
        label: str = "",
        point_color: Tuple[int, int, int] = (255, 0, 0),
        point_radius_pct: float = 0.01  # 0.5% of image width
    ) -> Image.Image:
        """Create an overlay image with point markers.
        
        Uses the same point style as SAM2 for consistency.
        
        Args:
            image: Original PIL image
            points: List of (x, y) pixel coordinates to mark
            label: Optional label for the detection (used in future enhancements)
            point_color: RGB color for the point markers
            point_radius_pct: Point marker radius as percentage of image width
            
        Returns:
            PIL Image with point markers overlaid
        """
        # Calculate actual pixel size from percentage
        point_radius = max(3, int(image.width * point_radius_pct))
        
        # Convert to RGBA for transparency support
        overlay = image.convert("RGBA")
        
        # Draw point markers
        draw = ImageDraw.Draw(overlay)
        # Point outline width scales with point size (about 33% of radius)
        point_outline_width = max(1, int(point_radius * 0.33))
        
        for x, y in points:
            # Draw a circle at each point (same style as SAM2)
            draw.ellipse(
                [x - point_radius, y - point_radius, 
                 x + point_radius, y + point_radius],
                fill=point_color,
                outline=(255, 255, 255),  # White outline for visibility
                width=point_outline_width
            )
        
        # Convert back to RGB (removing alpha channel)
        return overlay.convert("RGB")
    
    @staticmethod
    def _resolve_image(image):
        """Convert to PIL.Image if necessary."""
        if not isinstance(image, Image.Image):
            # Best-effort conversion for numpy arrays / bytes.
            try:
                image = Image.fromarray(image)  # type: ignore[arg-type]
            except Exception:
                raise TypeError(
                    "Unsupported image type for VLMTool; expected PIL.Image "
                    "or numpy array.")
        
        # # DEBUG: Final check - ensure we have a valid PIL.Image before returning
        # if not isinstance(image, Image.Image):
        #     print(f"[VLM_DEBUG] ERROR: _resolve_image returning non-PIL object: {type(image)}")
        #     exit()
        # else:
        #     print(f"[VLM_DEBUG] SUCCESS: _resolve_image returning valid PIL.Image")
            
        return image

    def _run_llm(self, image: Image.Image, prompt: str) -> str:
        """Execute the multimodal prompt and return raw text output."""
        
        # # DEBUG: Check if image parameter is valid PIL.Image at start of _run_llm
        # if not isinstance(image, Image.Image):
        #     print(f"[VLM_DEBUG] ERROR: _run_llm received non-PIL object: {type(image)}")
        #     exit()
        # else:
        #     print(f"[VLM_DEBUG] SUCCESS: _run_llm received valid PIL.Image")
        
        print(f"VLM prompt:\n{prompt}")

        # Process the image and text
        inputs = self._processor.process(
            images=[image],
            text=prompt
        )

        # Get model device and ensure it's valid
        if hasattr(self._model, 'device'):
            model_device = self._model.device
        else:
            # Fallback: get device from model parameters
            model_device = next(self._model.parameters()).device
        
        print(f"Model device: {model_device}")

        # Move inputs to the correct device and make a batch of size 1
        try:
            # Check if tensors can be moved to device safely
            inputs_on_device = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    print(f"Moving tensor {k} with shape {v.shape} from {v.device} to {model_device}")
                    # Ensure tensor is contiguous before moving
                    if not v.is_contiguous():
                        v = v.contiguous()
                    # Validate device compatibility before transfer
                    if v.device.type != model_device.type:
                        print(f"Device type mismatch: tensor on {v.device.type}, model on {model_device.type}")
                    inputs_on_device[k] = v.to(model_device, non_blocking=True).unsqueeze(0)
                else:
                    inputs_on_device[k] = v
            inputs = inputs_on_device
        except RuntimeError as e:
            print(f"Error moving tensors to device {model_device}: {e}")
            print("Attempting recovery with memory cleanup...")
            # Only clear cache in error recovery
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Try once more with synchronous transfer
            try:
                inputs_on_device = {}
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs_on_device[k] = v.to(model_device, non_blocking=False).unsqueeze(0)
                    else:
                        inputs_on_device[k] = v
                inputs = inputs_on_device
                print("Recovery successful")
            except RuntimeError as e2:
                print(f"Recovery failed: {e2}")
                raise e2

        # Generate output with proper parameters
        with torch.no_grad():
            output = self._model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=True,
                    stop_strings=["<|endoftext|>"]
                ),
                tokenizer=self._processor.tokenizer
            )

        # Only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self._processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        
        print(f"VLM response:\n{generated_text}")
        return generated_text

    @staticmethod
    def _parse_boxes(text: str) -> List[BoundingBox]:
        """Extract bounding boxes from Molmo's textual response."""
        text = text.strip()
        # Try strict JSON first.
        try:
            boxes = json.loads(text)
            if isinstance(boxes, list) and all(
                isinstance(b, list) and len(b) == 4 for b in boxes):
                return [tuple(map(int, b)) for b in boxes]  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

        # Fallback: regex for sequences of four ints.
        nums = list(map(int, re.findall(r"[-+]?[0-9]+", text)))
        boxes = [tuple(nums[i:i + 4]) for i in range(0, len(nums), 4)]
        return boxes

    @staticmethod
    def _parse_points(text: str) -> List[Point]:
        """Extract (x, y) points from Molmo's <point> / <points> response."""
        text = text.strip()
        points: List[Point] = []

        # Try to find both single point and multiple points patterns
        # Pattern 1: <point x="..." y="..." alt="...">...</point>
        single_point_matches = re.findall(r'<point\s+x="([0-9.+-]+)"\s+y="([0-9.+-]+)"[^>]*>', text)
        for x_str, y_str in single_point_matches:
            try:
                points.append((float(x_str), float(y_str)))
            except ValueError:
                continue

        # Pattern 2: <points x1="..." y1="..." x2="..." y2="..." ...>...</points>
        points_match = re.search(r"<points\s+([^>]+)>", text)
        if points_match:
            attrs = points_match.group(1)
            # Find all numeric values that appear in x / y attributes (order preserved)
            values = re.findall(r"(?:x\d*|y\d*)=\"([0-9.+-]+)\"", attrs)
            coords = []
            for val in values:
                try:
                    coords.append(float(val))
                except ValueError:
                    continue
            
            # Pair them (x1,y1,x2,y2,...) -> [(x1,y1), (x2,y2), ...]
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    points.append((coords[i], coords[i + 1]))

        points = [(x / 100.0, y / 100.0) for x, y in points] # return normalized points in [0, 1]
        return points 