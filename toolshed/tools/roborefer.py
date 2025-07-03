# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

"""RoboRefer vision-language tool backed by the RoboRefer-8B Llava checkpoint.

This tool mirrors the public interface of ``vlm.VLMTool`` so that controller
code can switch between the two models without modification.

The implementation relies on the *llava* package as used by the upstream
``simple_inference.py`` script that ships with the RoboRefer repository.  The
actor therefore has to run inside the *roborefer* Conda environment created by
that repository or at least have ``llava`` and the RoboRefer checkpoint
installed.

Example *tool_configs* excerpt for launching the tool on a single GPU:

    tool_configs = {
        "roborefer": {
            "cls": "toolshed.tools.roborefer.RoboreferTool",
            "actor_options": {"num_gpus": 1},
            "init_args": {
                "model_path": "/lustre/fsw/portfolios/nvr/users/siyic/projects/RoboRefer/models/RoboRefer-8B-SFT",
                "precision": "fp16"
            }
        }
    }
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import List, Tuple

import ray
from PIL import Image, ImageDraw

from toolshed.tools.base import BaseTool, tool_method
from toolshed.tool_result import ToolResult

# Heavy dependencies - imported directly since tools are only instantiated
# inside Ray actors running in tool-specific conda environments where
# dependencies are guaranteed to be available.
import torch  # type: ignore
import llava  # type: ignore
from llava.media import Image as LlavaImage

logger = logging.getLogger(__name__)


def _find_toolshed_root() -> str:
    """Find toolshed root from package location."""
    import toolshed
    return os.path.dirname(os.path.dirname(os.path.abspath(toolshed.__file__)))


BoundingBox = Tuple[int, int, int, int]
Point = Tuple[float, float]


class RoboreferTool(BaseTool):
    """Vision-Language model tool using the RoboRefer checkpoint via Llava."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        precision: str = "fp16",
        dtype: str | None = None,
        no_output_image: bool = False,
        no_output_vars: bool = False,
        exclude_methods: list[str] | None = None,
        exclude_behavior: str = "warn",
    ) -> None:
        super().__init__(
            no_output_image=no_output_image,
            no_output_vars=no_output_vars,
            exclude_methods=exclude_methods,
            exclude_behavior=exclude_behavior,
        )

        print("WARNING: RoboRefer forcing no_output_vars to True")
        self.no_output_vars = True

        # Resolve checkpoint path and validate
        model_path_obj = Path(model_path).expanduser()
        # If relative path, resolve relative to toolshed root
        if not model_path_obj.is_absolute():
            toolshed_root = _find_toolshed_root()
            model_path_obj = Path(toolshed_root) / model_path_obj
        self._model_path = model_path_obj.resolve()
        if not self._model_path.exists():
            raise FileNotFoundError(f"RoboRefer checkpoint not found: {self._model_path}")

        # Map precision argument to torch dtype – allow explicit override via dtype,
        # but keep *precision* for parity with simple_inference.py.
        if dtype is None:
            precision_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            dtype = precision_map.get(precision, torch.float16)  # default to fp16
        self._dtype = dtype  # store for debugging

        logger.info("Loading RoboRefer model from %s (dtype=%s) …", self._model_path, dtype)
        load_kwargs = {
            "torch_dtype": dtype,
        } if dtype != torch.float32 else {}

        # Expensive model load happens once per actor – keep reference.
        self._model = llava.load(str(self._model_path), **load_kwargs)

    # ------------------------------------------------------------------
    # BaseTool interface implementation
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        return "roborefer"

    # ------------------------------------------------------------------
    # Public API methods (identical surface to VLMTool)
    # ------------------------------------------------------------------

    @tool_method
    def detect_one(self, image, obj_name: str) -> ToolResult[List[Point]]:  # noqa: D401
        """Detect *one* instance of *obj_name* in *image*.

        [[if:text]]Text output: coordinates of a single point for the first instance of the object, in normalized pixel space in range [0, 1].[[/if:text]]
        [[if:image]]Visual output: an overlay image with detected object marked as red point.[[/if:image]]
        [[if:vars]]Stored variables: Coordinates in $<obj_name>_detections variable of list of (x, y) point coordinates in normalized pixel space [0, 1] for use in subsequent operations.[[/if:vars]]
        """
        img = self._resolve_image(image)
        response_text = self._run_llm(img, f"Point to the {obj_name} in this image.")
        points = self._parse_points(response_text)
        if points:
            points = [points[0]]  # only first point
        return self._detection_result(img, points, obj_name)

    @tool_method
    def detect_all(self, image, obj_name: str) -> ToolResult[List[Point]]:  # noqa: D401
        """Detect *all* instances of *obj_name* in *image*.

        [[if:text]]Text output: List of point coordinates for the detected objects, in normalized pixel space in range [0, 1].[[/if:text]]
        [[if:image]]Visual output: an overlay image with detected objects marked as red points.[[/if:image]]
        [[if:vars]]Stored variables: Coordinates in $<obj_name>_detections variable of list of (x, y) point coordinates in normalized pixel space [0, 1] for use in subsequent operations.[[/if:vars]]
        
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

    #@tool_method – optional free-form question answering (disabled to match VLMTool)
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
        answer = self._run_llm(img, query)
        answer_text = answer.strip()
        return ToolResult(answer_text, text=answer_text)

    # ------------------------------------------------------------------
    # Internal helpers (mostly copied from VLMTool)
    # ------------------------------------------------------------------

    def _detection_result(self, img: Image.Image, points: List[Point], obj_name: str) -> ToolResult[List[Point]]:
        points_str = str(points)
        num_detected = len(points)

        # Create overlay visualisation if desired
        overlay_image = None
        if points and not self.no_output_image:
            w, h = img.size
            pixel_points = [(int(x * w), int(y * h)) for x, y in points]
            overlay_image = self._create_point_overlay(img, pixel_points, obj_name)

        safe_var_name = f"{obj_name.replace(' ', '_')}_detections"
        variables = {} if self.no_output_vars else ({safe_var_name: points} if points else {})

        text = f"Detected {num_detected} instance(s) of '{obj_name}': {points_str}"
        if points and not self.no_output_vars:
            text += f" Use ${safe_var_name} to reference them."

        return ToolResult(points, text=text, image=overlay_image, variables=variables)

    @staticmethod
    def _resolve_image(image):
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)  # type: ignore[arg-type]
            except Exception:
                raise TypeError(
                    "Unsupported image type for RoboreferTool; expected PIL.Image or numpy array."
                )
        return image

    @staticmethod
    def _create_point_overlay(
        image: Image.Image,
        points: List[Tuple[int, int]],
        label: str = "",
        point_color: Tuple[int, int, int] = (255, 0, 0),
        point_radius_pct: float = 0.01,  # 1% of image width
    ) -> Image.Image:
        """Create an overlay image marking *points* with coloured circles."""
        point_radius = max(3, int(image.width * point_radius_pct))
        overlay = image.convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        outline_width = max(1, int(point_radius * 0.33))
        for x, y in points:
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=point_color,
                outline=(255, 255, 255),
                width=outline_width,
            )
        return overlay.convert("RGB")

    def _run_llm(self, image: Image.Image, prompt: str) -> str:
        """Run the RoboRefer model on (*image*, *prompt*) and return raw text."""
        logger.debug("RoboRefer prompt: %s", prompt)

        # Ensure image is in RGB mode (JPEG doesn't support RGBA)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir="/dev/shm") as tmp:
            image.save(tmp, format="JPEG", quality=95)
            tmp_path = Path(tmp.name)

        # Always append formatting hint so the model returns a list of tuples
        formatting_hint = (
            "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)],  "
            "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
            "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
        )
        full_prompt = f"{prompt} {formatting_hint}"

        prompt_media = [LlavaImage(str(tmp_path)), full_prompt]

        try:
            with torch.no_grad():
                answer = self._model.generate_content(prompt_media)
        finally:
            # Clean up temporary file immediately
            tmp_path.unlink(missing_ok=True)

        logger.debug("RoboRefer response: %s", answer)
        return answer

    # Robust point parser – RoboRefer already returns coordinates in [0,1]
    @staticmethod
    def _parse_points(text: str) -> List[Point]:
        """Extract list of (x, y) tuples from RoboRefer textual output."""
        import ast
        text = text.strip()
        # 1) Try Python literal eval for list-of-tuples representation
        try:
            pts = ast.literal_eval(text)
            if isinstance(pts, list):
                result: List[Point] = []
                for itm in pts:
                    if (
                        isinstance(itm, (list, tuple))
                        and len(itm) == 2
                        and all(isinstance(v, (int, float)) for v in itm)
                    ):
                        result.append((float(itm[0]), float(itm[1])))
                if result:
                    return result
        except Exception:
            pass
        # 2) Fallback regex numbers pairing
        nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", text)))
        return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
