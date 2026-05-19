# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Projection dome sky-marking panel."""

import json
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import lichtfeld as lf
from . import rml_widgets as w
from .types import Panel

__lfs_panel_classes__ = ["SkyMarkerPanel"]
__lfs_panel_ids__ = ["lfs.sky_marker"]

_instance = None
_RML_PATH_SAFE_CHARS = "/:._-~"
ZOOM_MIN = 0.5
ZOOM_MAX = 8.0

FACE_IDS = ("neg_y", "neg_x", "pos_z", "pos_x", "neg_z", "pos_y")
FACE_LABELS = {
    "pos_x": "+X",
    "neg_x": "-X",
    "pos_y": "+Y",
    "neg_y": "-Y",
    "pos_z": "+Z",
    "neg_z": "-Z",
}


def _encode_rml_path(path: Path | str, drive_source: Path | str | None = None) -> str:
    text = str(path).replace("\\", "/")
    if (
        text.startswith("/")
        and not text.startswith("//")
        and not (len(text) >= 3 and text[0].isalpha() and text[1] == ":")
        and drive_source is not None
    ):
        drive_text = str(drive_source).replace("\\", "/")
        if len(drive_text) >= 3 and drive_text[0].isalpha() and drive_text[1] == ":":
            text = drive_text[:2] + text
    return quote(text, safe=_RML_PATH_SAFE_CHARS)


def _absolute_path(path: Path | str, base: Path | None = None) -> Path:
    value = Path(str(path)).expanduser()
    if not value.is_absolute():
        value = (base or Path.cwd()) / value
    return value.resolve(strict=False)


def _optional_absolute_path(path, base: Path | None = None) -> Path | None:
    text = str(path or "").strip()
    if not text:
        return None
    return _absolute_path(text, base)


def open_sky_marker_panel(reset_mask: bool = True) -> bool:
    lf.ui.set_panel_enabled("lfs.sky_marker", True)
    if _instance is None:
        return False
    _instance.open(reset_mask=reset_mask)
    return True


class SkyMarkerPanel(Panel):
    id = "lfs.sky_marker"
    label = "Sky Marker"
    space = lf.ui.PanelSpace.FLOATING
    order = 94
    template = "rmlui/sky_marker_panel.rml"
    size = (1280, 860)
    update_interval_ms = 33

    def __init__(self):
        global _instance
        _instance = self
        self._doc = None
        self._handle = None
        self._workspace = None
        self._manifest_path = None
        self._dome_world = None
        self._face_size = 512
        self._brush_radius = 24
        self._zoom = 1.25
        self._mode = "paint"
        self._active_face = "pos_z"
        self._painting = False
        self._states = {}
        self._status = ""
        self._status_kind = ""
        self._status_time = 0.0
        self._overlay_revision = 0
        self._pending_overlay_faces = set()
        self._last_overlay_flush_time = 0.0
        self._manifest_dirty = False
        self._pending_open = False
        self._pending_reset = False

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("sky_marker")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("sky_marker.title"))
        model.bind_func("paint_mode", lambda: self._mode == "paint")
        model.bind_func("erase_mode", lambda: self._mode == "erase")
        model.bind_func("brush_radius_text", lambda: str(self._brush_radius))
        model.bind_func("zoom_text", lambda: f"{self._zoom * 100.0:.0f}%")
        model.bind_func("active_face_id", lambda: self._active_face)
        model.bind_func("active_face_label", lambda: FACE_LABELS.get(self._active_face, self._active_face))
        model.bind_func("active_face_size_style", self._active_face_size_style)
        model.bind_func("active_preview", lambda: self._decorator(self._active_face, "preview_path"))
        model.bind_func("active_overlay", lambda: self._decorator(self._active_face, "overlay_display_path"))
        model.bind_func("active_count", lambda: self._count_label(self._active_face))
        model.bind_func("active_disabled", lambda: self._is_disabled(self._active_face))
        model.bind_func("status_text", lambda: self._visible_status())
        model.bind_func("status_error", lambda: self._status_visible() and self._status_kind == "error")
        model.bind_func("status_success", lambda: self._status_visible() and self._status_kind == "success")
        model.bind_func("workspace_text", self._workspace_text)

        for face_id in FACE_IDS:
            model.bind_func(f"preview_{face_id}", lambda fid=face_id: self._decorator(fid, "preview_path"))
            model.bind_func(f"overlay_{face_id}", lambda fid=face_id: self._decorator(fid, "overlay_display_path"))
            model.bind_func(f"count_{face_id}", lambda fid=face_id: self._count_label(fid))
            model.bind_func(f"selected_{face_id}", lambda fid=face_id: self._active_face == fid)
            model.bind_func(f"disabled_{face_id}", lambda fid=face_id: self._is_disabled(fid))

        model.bind_event("action", self._on_action)
        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        cube = doc.get_element_by_id("cubemap")
        if cube:
            cube.add_event_listener("mousedown", self._on_cubemap_mousedown)
        active = doc.get_element_by_id("active-face")
        if active:
            active.add_event_listener("mousedown", self._on_face_mousedown)
            active.add_event_listener("mousemove", self._on_face_mousemove)
            active.add_event_listener("mouseup", self._on_face_mouseup)
            active.add_event_listener("mouseout", self._on_face_mouseup)
        active_scroll = doc.get_element_by_id("active-scroll")
        if active_scroll:
            active_scroll.add_event_listener("mousescroll", self._on_zoom_scroll)
        doc.add_event_listener("mouseup", self._on_face_mouseup)
        if self._pending_open:
            reset = self._pending_reset
            self._pending_open = False
            self._pending_reset = False
            self.open(reset_mask=reset)

    def on_update(self, doc):
        del doc
        return False

    def open(self, reset_mask: bool = True):
        if self._doc is None:
            self._pending_open = True
            self._pending_reset = reset_mask
            return

        try:
            lf.ensure_projection_dome()
            self._workspace = self._default_workspace()
            self._manifest_path = self._workspace / "sky_mask_manifest.json"
            self._pending_overlay_faces.clear()
            self._manifest_dirty = False
            if reset_mask:
                self._clear_display_overlay_cache()
            result = lf.prepare_projection_dome_sky_cubemap(
                str(self._workspace),
                face_size=self._face_size,
                overwrite_preview=True,
                reset_mask=reset_mask,
            )
            dome_world = result.get("dome_world")
            if isinstance(dome_world, (list, tuple)) and len(dome_world) == 16:
                self._dome_world = [float(value) for value in dome_world]
            else:
                self._dome_world = None
            self._states = {}
            for face in result.get("faces", []):
                face_id = str(face.get("id", ""))
                if face_id not in FACE_IDS:
                    continue
                self._states[face_id] = {
                    "id": face_id,
                    "label": str(face.get("label", FACE_LABELS.get(face_id, face_id))),
                    "preview_path": _optional_absolute_path(face.get("preview_path"), self._workspace),
                    "mask_path": _optional_absolute_path(face.get("mask_path"), self._workspace),
                    "overlay_path": _optional_absolute_path(face.get("overlay_path"), self._workspace),
                    "overlay_display_path": None,
                    "valid_pixels": int(face.get("valid_pixels", 0)),
                    "marked_pixels": int(face.get("marked_pixels", 0)),
                }
                self._sync_display_overlay(face_id)

            self._write_manifest()
            self._manifest_dirty = False
            self._set_status("success", lf.ui.tr("sky_marker.status_ready"))
        except Exception as exc:
            self._set_status("error", f"{lf.ui.tr('sky_marker.status_failed')}: {exc}")

        if self._handle:
            self._handle.dirty_all()

    def _default_workspace(self) -> Path:
        dataset = lf.dataset_params()
        if dataset and dataset.has_params():
            if dataset.output_path:
                output = _absolute_path(dataset.output_path)
            elif dataset.data_path:
                output = _absolute_path(dataset.data_path) / "output"
            else:
                output = _absolute_path("output")
            return output / "projection_dome" / "sky_cubemap"
        return _absolute_path("projection_dome") / "sky_cubemap"

    def _decorator(self, face_id: str, key: str) -> str:
        state = self._states.get(face_id)
        if not state:
            return "none"
        path = state.get(key)
        if not path:
            return "none"
        path = Path(path)
        if not path.exists():
            return "none"
        return f"image({_encode_rml_path(path, self._workspace)})"

    def _active_face_size_style(self) -> str:
        size = max(1, int(round(self._face_size * self._zoom)))
        return f"{size}dp"

    def _count_label(self, face_id: str) -> str:
        state = self._states.get(face_id)
        if not state:
            return ""
        marked = int(state.get("marked_pixels", 0))
        valid = max(1, int(state.get("valid_pixels", 0)))
        pct = min(100.0, marked * 100.0 / valid)
        return f"{pct:.1f}%"

    def _is_disabled(self, face_id: str) -> bool:
        state = self._states.get(face_id)
        return bool(state and int(state.get("valid_pixels", 0)) <= 0)

    def _workspace_text(self) -> str:
        if not self._manifest_path:
            return ""
        return str(self._manifest_path)

    def _status_visible(self) -> bool:
        return bool(self._status) and time.time() - self._status_time < 6.0

    def _visible_status(self) -> str:
        return self._status if self._status_visible() else ""

    def _set_status(self, kind: str, text: str):
        self._status_kind = kind
        self._status = text
        self._status_time = time.time()
        if self._handle:
            self._handle.dirty("status_text")
            self._handle.dirty("status_error")
            self._handle.dirty("status_success")

    def _on_action(self, handle, event, args):
        del handle, event
        if not args:
            return
        action = str(args[0])
        if action == "paint":
            self._mode = "paint"
            self._dirty_modes()
        elif action == "erase":
            self._mode = "erase"
            self._dirty_modes()
        elif action == "brush_down":
            self._brush_radius = max(2, self._brush_radius - 4)
            self._dirty_brush()
        elif action == "brush_up":
            self._brush_radius = min(128, self._brush_radius + 4)
            self._dirty_brush()
        elif action == "zoom_out":
            self._set_zoom(self._zoom / 1.25)
        elif action == "zoom_in":
            self._set_zoom(self._zoom * 1.25)
        elif action == "zoom_reset":
            self._set_zoom(1.0)
        elif action == "refresh":
            self.open(reset_mask=False)
        elif action == "clear_face":
            self._clear_face(self._active_face)
        elif action == "clear_all":
            for face_id in FACE_IDS:
                self._clear_face(face_id, quiet=True)
            self._write_manifest()
            self._manifest_dirty = False
            self._set_status("success", lf.ui.tr("sky_marker.status_cleared"))
            if self._handle:
                self._handle.dirty_all()
        elif action == "save":
            if self._save_for_training():
                lf.ui.set_panel_enabled(self.id, False)

    def _dirty_modes(self):
        if self._handle:
            self._handle.dirty("paint_mode")
            self._handle.dirty("erase_mode")

    def _dirty_brush(self):
        if self._handle:
            self._handle.dirty("brush_radius_text")

    def _dirty_active_face(self):
        if not self._handle:
            return
        for fid in FACE_IDS:
            self._handle.dirty(f"selected_{fid}")
        self._handle.dirty("active_face_id")
        self._handle.dirty("active_face_label")
        self._handle.dirty("active_preview")
        self._handle.dirty("active_overlay")
        self._handle.dirty("active_count")
        self._handle.dirty("active_disabled")

    def _dirty_zoom(self):
        if self._handle:
            self._handle.dirty("zoom_text")
            self._handle.dirty("active_face_size_style")

    def _set_zoom(self, value: float):
        next_zoom = max(ZOOM_MIN, min(ZOOM_MAX, float(value)))
        if abs(next_zoom - self._zoom) < 0.001:
            return
        self._zoom = next_zoom
        self._dirty_zoom()

    def _set_active_face(self, face_id: str):
        if face_id not in FACE_IDS or face_id == self._active_face:
            return
        self._active_face = face_id
        self._dirty_active_face()

    def _on_cubemap_mousedown(self, event):
        cube = self._doc.get_element_by_id("cubemap") if self._doc else None
        face_el = w.find_ancestor_with_attribute(event.target(), "data-face-id", cube)
        if not face_el:
            return
        face_id = face_el.get_attribute("data-face-id", "")
        if face_id in self._states:
            self._set_active_face(face_id)
            event.stop_propagation()

    def _on_zoom_scroll(self, event):
        try:
            delta = float(event.get_parameter("wheel_delta_y", "0"))
        except (TypeError, ValueError):
            return
        if delta == 0.0:
            return
        self._set_zoom(self._zoom * 1.12 if delta > 0.0 else self._zoom / 1.12)
        event.stop_propagation()

    def _on_face_mousedown(self, event):
        self._painting = self._paint_from_event(event)
        if self._painting:
            event.stop_propagation()

    def _on_face_mousemove(self, event):
        if self._painting:
            if self._paint_from_event(event):
                event.stop_propagation()

    def _on_face_mouseup(self, event):
        del event
        was_painting = self._painting
        self._painting = False
        self._flush_pending_overlays(force=True)
        if was_painting and self._manifest_dirty:
            self._write_manifest()
            self._manifest_dirty = False

    def _paint_from_event(self, event):
        root = self._doc.get_element_by_id("active-scroll") if self._doc else None
        face_el = w.find_ancestor_with_attribute(event.target(), "data-face-id", root)
        if not face_el:
            return False
        face_id = face_el.get_attribute("data-face-id", "")
        if face_id not in self._states or self._is_disabled(face_id):
            return False

        self._set_active_face(face_id)

        try:
            mx = float(event.get_parameter("mouse_x", "0"))
            my = float(event.get_parameter("mouse_y", "0"))
        except ValueError:
            return False

        width = max(1.0, float(face_el.absolute_width))
        height = max(1.0, float(face_el.absolute_height))
        lx = mx - float(face_el.absolute_left)
        ly = my - float(face_el.absolute_top)
        if lx < 0.0 or ly < 0.0 or lx > width or ly > height:
            return False

        x = int(max(0, min(self._face_size - 1, round(lx / width * (self._face_size - 1)))))
        y = int(max(0, min(self._face_size - 1, round(ly / height * (self._face_size - 1)))))
        button = event.get_parameter("button", "0")
        erase = self._mode == "erase" or button in ("1", "2")
        self._stroke(face_id, x, y, erase)
        return True

    def _stroke(self, face_id: str, x: int, y: int, erase: bool):
        state = self._states.get(face_id)
        if not state:
            return
        try:
            result = lf.paint_sky_cubemap_mask(
                str(state["mask_path"]),
                str(state["overlay_path"]),
                self._face_size,
                x,
                y,
                self._brush_radius,
                erase=erase,
            )
            state["marked_pixels"] = int(result.get("marked_pixels", 0))
            self._schedule_overlay_sync(face_id)
            self._manifest_dirty = True
            if self._handle:
                self._handle.dirty(f"count_{face_id}")
                if face_id == self._active_face:
                    self._handle.dirty("active_count")
                self._handle.dirty("workspace_text")
        except Exception as exc:
            self._set_status("error", f"{lf.ui.tr('sky_marker.status_failed')}: {exc}")

    def _clear_face(self, face_id: str, quiet: bool = False):
        state = self._states.get(face_id)
        if not state:
            return
        try:
            result = lf.clear_sky_cubemap_mask(
                str(state["mask_path"]),
                str(state["overlay_path"]),
                self._face_size,
            )
            state["marked_pixels"] = int(result.get("marked_pixels", 0))
            self._pending_overlay_faces.discard(face_id)
            self._sync_display_overlay(face_id)
            if not quiet:
                self._write_manifest()
                self._manifest_dirty = False
                self._set_status("success", lf.ui.tr("sky_marker.status_cleared"))
            if self._handle:
                self._handle.dirty(f"overlay_{face_id}")
                self._handle.dirty(f"count_{face_id}")
                if face_id == self._active_face:
                    self._handle.dirty("active_overlay")
                    self._handle.dirty("active_count")
        except Exception as exc:
            self._set_status("error", f"{lf.ui.tr('sky_marker.status_failed')}: {exc}")

    def _clear_display_overlay_cache(self):
        # RmlUI can still hold decorators that reference older overlay files.
        # Keep the cache during the panel lifetime; reset_mask clears the actual mask.
        return

    def _schedule_overlay_sync(self, face_id: str):
        self._pending_overlay_faces.add(face_id)

    def _flush_pending_overlays(self, force: bool = False):
        if not self._pending_overlay_faces:
            return
        if not force:
            return
        now = time.time()
        faces = tuple(self._pending_overlay_faces)
        self._pending_overlay_faces.clear()
        self._last_overlay_flush_time = now
        for face_id in faces:
            self._sync_display_overlay(face_id)
            if self._handle:
                self._handle.dirty(f"overlay_{face_id}")
                if face_id == self._active_face:
                    self._handle.dirty("active_overlay")

    def _sync_display_overlay(self, face_id: str):
        state = self._states.get(face_id)
        if not state:
            return
        overlay = Path(state["overlay_path"])
        if not overlay.exists():
            return
        self._overlay_revision += 1
        display = overlay.with_name(f"{overlay.stem}_ui_{self._overlay_revision}{overlay.suffix}")
        try:
            shutil.copyfile(overlay, display)
            state["overlay_display_path"] = display
        except OSError:
            state["overlay_display_path"] = overlay

    def _write_manifest(self):
        if not self._manifest_path:
            return
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        faces = {}
        for face_id in FACE_IDS:
            state = self._states.get(face_id)
            if not state:
                continue
            faces[face_id] = {
                "label": state.get("label", FACE_LABELS.get(face_id, face_id)),
                "mask": str(state.get("mask_path", "")),
                "preview": str(state.get("preview_path", "")),
                "valid_pixels": int(state.get("valid_pixels", 0)),
                "marked_pixels": int(state.get("marked_pixels", 0)),
            }
        data = {
            "type": "projection_dome_sky_cubemap_mask",
            "face_size": self._face_size,
            "faces": faces,
        }
        if self._dome_world:
            data["dome_world"] = self._dome_world
        self._manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        params = lf.optimization_params()
        if params and params.has_params() and hasattr(params, "sky_mask_path"):
            params.sky_mask_path = str(self._manifest_path)

    def _save_for_training(self) -> bool:
        try:
            self._flush_pending_overlays(force=True)
            self._write_manifest()
            self._manifest_dirty = False
            result = lf.preview_projection_dome_sky_initialization(str(self._manifest_path))
            count = int(result.get("gaussian_count", 0))
            self._set_status(
                "success",
                f"{lf.ui.tr('sky_marker.status_saved')} {count:,} sky gaussians ready.",
            )
            return True
        except Exception as exc:
            self._set_status("error", f"{lf.ui.tr('sky_marker.status_failed')}: {exc}")
            return False
