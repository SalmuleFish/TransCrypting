"""Desktop launcher for TransCrypting (100% local audio transcription, summary & chat)."""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from backend.app import app
from backend.config import DEFAULT_PORT, FRONTEND_DIR, HOST
from backend.database import db

ICON_PATH = FRONTEND_DIR / "icon.ico"
APP_USER_MODEL_ID = "TransCrypting.LocalApp"
_CACHED_HICON_SM = 0
_CACHED_HICON_LG = 0


def _rgb_to_colorref(r: int, g: int, b: int) -> int:
    return (b << 16) | (g << 8) | r


def _set_process_app_user_model_id() -> None:
    """Associate all Windows UI/taskbar surfaces of this process with TransCrypting."""
    if sys.platform != "win32":
        return
    try:
        from ctypes import wintypes

        shell32 = ctypes.windll.shell32
        shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [wintypes.LPCWSTR]
        shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
        shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def _load_windows_icons(icon_path: Path) -> tuple[int, int]:
    """Load and cache 64-bit-safe small and large HICON handles from icon.ico."""
    global _CACHED_HICON_SM, _CACHED_HICON_LG
    if _CACHED_HICON_SM and _CACHED_HICON_LG:
        return _CACHED_HICON_SM, _CACHED_HICON_LG
    if not icon_path.exists():
        return 0, 0

    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.LoadImageW.argtypes = [
        wintypes.HINSTANCE,
        wintypes.LPCWSTR,
        wintypes.UINT,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.LoadImageW.restype = wintypes.HANDLE

    IMAGE_ICON = 1
    LR_LOADFROMFILE = 0x0010
    cx_sm = max(16, int(user32.GetSystemMetrics(49) or 16))
    cy_sm = max(16, int(user32.GetSystemMetrics(50) or 16))
    cx_lg = max(48, int(user32.GetSystemMetrics(11) or 32))
    cy_lg = max(48, int(user32.GetSystemMetrics(12) or 32))

    icon_str = str(icon_path.resolve())
    hicon_sm = user32.LoadImageW(None, icon_str, IMAGE_ICON, cx_sm, cy_sm, LR_LOADFROMFILE)
    hicon_lg = user32.LoadImageW(None, icon_str, IMAGE_ICON, cx_lg, cy_lg, LR_LOADFROMFILE)
    if not hicon_lg:
        hicon_lg = user32.LoadImageW(None, icon_str, IMAGE_ICON, 32, 32, LR_LOADFROMFILE)
    if not hicon_sm:
        hicon_sm = user32.LoadImageW(None, icon_str, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)

    _CACHED_HICON_SM = int(hicon_sm or 0)
    _CACHED_HICON_LG = int(hicon_lg or 0)
    return _CACHED_HICON_SM, _CACHED_HICON_LG


def _set_windows_taskbar_properties(hwnd: int, icon_path: Path) -> bool:
    """Bind per-window AppUserModelID and relaunch icon/command properties for the Windows 11 taskbar."""
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        class PROPERTYKEY(ctypes.Structure):
            _fields_ = [
                ("fmtid", GUID),
                ("pid", wintypes.DWORD),
            ]

        class PROPVARIANT(ctypes.Structure):
            _fields_ = [
                ("vt", ctypes.c_ushort),
                ("wReserved1", ctypes.c_ushort),
                ("wReserved2", ctypes.c_ushort),
                ("wReserved3", ctypes.c_ushort),
                ("pwszVal", ctypes.c_void_p),
                ("pad", ctypes.c_uint64),
            ]

        ole32 = ctypes.windll.ole32
        shell32 = ctypes.windll.shell32

        ole32.CLSIDFromString.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(GUID)]
        ole32.CLSIDFromString.restype = ctypes.c_long
        ole32.CoTaskMemAlloc.argtypes = [ctypes.c_size_t]
        ole32.CoTaskMemAlloc.restype = ctypes.c_void_p
        ole32.PropVariantClear.argtypes = [ctypes.POINTER(PROPVARIANT)]
        ole32.PropVariantClear.restype = ctypes.c_long

        shell32.SHGetPropertyStoreForWindow.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        shell32.SHGetPropertyStoreForWindow.restype = ctypes.c_long

        iid_prop_store = GUID()
        fmtid_app_user_model = GUID()
        if ole32.CLSIDFromString("{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}", ctypes.byref(iid_prop_store)) < 0:
            return False
        if ole32.CLSIDFromString("{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}", ctypes.byref(fmtid_app_user_model)) < 0:
            return False

        store_ptr = ctypes.c_void_p()
        hr = shell32.SHGetPropertyStoreForWindow(
            wintypes.HWND(hwnd),
            ctypes.byref(iid_prop_store),
            ctypes.byref(store_ptr),
        )
        if hr < 0 or not store_ptr.value:
            return False

        vtable = ctypes.cast(
            ctypes.cast(store_ptr.value, ctypes.POINTER(ctypes.c_void_p))[0],
            ctypes.POINTER(ctypes.c_void_p),
        )
        release_fn = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
        set_value_fn = ctypes.WINFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.POINTER(PROPERTYKEY),
            ctypes.POINTER(PROPVARIANT),
        )(vtable[6])
        commit_fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(vtable[7])

        app_dir = Path(__file__).resolve().parent
        exe_path = app_dir / "TransCrypting.exe"
        icon_resource = f"{exe_path},0" if exe_path.exists() else str(icon_path.resolve())
        relaunch_cmd = f'"{exe_path}"' if exe_path.exists() else f'"{sys.executable}" "{Path(__file__).resolve()}"'

        props = (
            (5, APP_USER_MODEL_ID),
            (3, icon_resource),
            (4, "TransCrypting"),
            (2, relaunch_cmd),
        )

        VT_LPWSTR = 31
        try:
            for pid, val in props:
                key = PROPERTYKEY(fmtid=fmtid_app_user_model, pid=pid)
                encoded = (val + "\0").encode("utf-16-le")
                mem = ole32.CoTaskMemAlloc(len(encoded))
                if not mem:
                    return False
                ctypes.memmove(mem, encoded, len(encoded))
                pv = PROPVARIANT(vt=VT_LPWSTR, pwszVal=mem)
                try:
                    set_hr = set_value_fn(store_ptr.value, ctypes.byref(key), ctypes.byref(pv))
                    if set_hr < 0:
                        return False
                finally:
                    ole32.PropVariantClear(ctypes.byref(pv))

            commit_hr = commit_fn(store_ptr.value)
            return commit_hr >= 0
        finally:
            release_fn(store_ptr.value)
    except Exception:
        return False


def _apply_windows_titlebar_style(
    window: object,
    icon_path: Path,
    theme: str = "light",
    apply_icon: bool = True,
) -> None:
    """Match Windows title bar colors to the active theme and apply the TC taskbar/window icons."""
    if sys.platform != "win32":
        return
    try:
        from ctypes import wintypes

        native = None
        for _ in range(40):
            native = getattr(window, "native", None)
            if native is not None:
                break
            time.sleep(0.05)
        if native is None:
            return

        handle_obj = native.Handle
        hwnd = (
            int(handle_obj.ToInt64())
            if hasattr(handle_obj, "ToInt64")
            else int(handle_obj.ToInt32())
        )

        is_dark = str(theme).lower() == "dark"
        dwmapi = ctypes.windll.dwmapi
        dwmapi.DwmSetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long

        if is_dark:
            caption_color = ctypes.c_int(_rgb_to_colorref(0x18, 0x1B, 0x1A))
            text_color = ctypes.c_int(_rgb_to_colorref(0xED, 0xF1, 0xEB))
            dark_mode = ctypes.c_int(1)
        else:
            caption_color = ctypes.c_int(_rgb_to_colorref(0xBC, 0xC5, 0xBD))
            text_color = ctypes.c_int(_rgb_to_colorref(0x12, 0x14, 0x13))
            dark_mode = ctypes.c_int(0)

        # 20: DWMWA_USE_IMMERSIVE_DARK_MODE, 34: BORDER, 35: CAPTION, 36: TEXT
        dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark_mode), ctypes.sizeof(dark_mode))
        dwmapi.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(caption_color), ctypes.sizeof(caption_color))
        dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(caption_color), ctypes.sizeof(caption_color))
        dwmapi.DwmSetWindowAttribute(hwnd, 36, ctypes.byref(text_color), ctypes.sizeof(text_color))

        if apply_icon and icon_path.exists():
            try:
                from System import Action
                from System.Drawing import Icon

                def _set_form_icon() -> None:
                    native.Icon = Icon(str(icon_path.resolve()))

                if getattr(native, "InvokeRequired", False):
                    native.Invoke(Action(_set_form_icon))
                else:
                    _set_form_icon()
            except Exception:
                pass

            hicon_sm, hicon_lg = _load_windows_icons(icon_path)
            user32 = ctypes.windll.user32
            user32.SendMessageW.argtypes = [
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.SendMessageW.restype = ctypes.c_ssize_t

            set_class_long = getattr(user32, "SetClassLongPtrW", None) or getattr(user32, "SetClassLongW")
            set_class_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
            set_class_long.restype = ctypes.c_void_p

            WM_SETICON = 0x0080
            GCLP_HICON = -14
            GCLP_HICONSM = -34

            if hicon_sm:
                user32.SendMessageW(hwnd, WM_SETICON, 0, hicon_sm)
                user32.SendMessageW(hwnd, WM_SETICON, 2, hicon_sm)
                set_class_long(hwnd, GCLP_HICONSM, ctypes.c_void_p(hicon_sm))
            if hicon_lg:
                user32.SendMessageW(hwnd, WM_SETICON, 1, hicon_lg)
                set_class_long(hwnd, GCLP_HICON, ctypes.c_void_p(hicon_lg))

            _set_windows_taskbar_properties(hwnd, icon_path)
    except Exception:
        pass


class DesktopBridge:
    """JS-to-Python bridge for native Windows save dialogs and titlebar theme sync."""

    def __init__(self) -> None:
        self._window: Any = None

    def attach(self, window: Any) -> None:
        self._window = window

    def save_markdown(self, filename: str, content: str) -> dict[str, Any]:
        """Open native Windows Save dialog and write UTF-8 Markdown file locally."""
        if self._window is None:
            return {"saved": False, "error": "Ventana nativa no disponible."}
        try:
            import webview

            dialog_type = getattr(
                getattr(webview, "FileDialog", None),
                "SAVE",
                getattr(webview, "SAVE_DIALOG", 30),
            )
            result = self._window.create_file_dialog(
                dialog_type,
                save_filename=filename or "transcripcion.md",
                file_types=("Markdown (*.md)", "Texto (*.txt)"),
            )
            if not result:
                return {"saved": False, "cancelled": True}

            target_path = result[0] if isinstance(result, (tuple, list)) else str(result)
            if not target_path:
                return {"saved": False, "cancelled": True}

            dest = Path(target_path)
            if not dest.suffix:
                dest = dest.with_suffix(".md")
            dest.write_text(content or "", encoding="utf-8")
            return {"saved": True, "path": str(dest)}
        except Exception as exc:
            return {"saved": False, "error": str(exc)}

    def set_titlebar_theme(self, theme: str) -> bool:
        """Synchronize the native Windows titlebar color with light/dark mode."""
        if self._window is None:
            return False
        _apply_windows_titlebar_style(self._window, ICON_PATH, theme=theme, apply_icon=False)
        return True


def find_available_port(start_port: int = DEFAULT_PORT) -> int:
    for port in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def wait_for_server(url: str, timeout_sec: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/api/health", timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def _launch_edge_app_fallback(url: str) -> bool:
    edge_candidates = [
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in edge_candidates:
        if path and os.path.exists(path):
            subprocess.Popen([path, f"--app={url}"])
            return True
    webbrowser.open(url)
    return False


def main() -> None:
    _set_process_app_user_model_id()

    parser = argparse.ArgumentParser(description="TransCrypting — Asistente Local de Audio")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Puerto local (127.0.0.1)")
    parser.add_argument("--browser", action="store_true", help="Abrir en el navegador en vez de ventana nativa")
    parser.add_argument("--headless", action="store_true", help="Iniciar solo el servidor local sin ventana")
    args = parser.parse_args()

    port = find_available_port(args.port)
    url = f"http://{HOST}:{port}"

    config = uvicorn.Config(
        app,
        host=HOST,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    if args.headless:
        print(f"TransCrypting escuchando en {url} (Ctrl+C para salir)")
        server.run()
        return

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    if not wait_for_server(url):
        print("Error: el servidor local de TransCrypting no respondió a tiempo.", file=sys.stderr)
        server.should_exit = True
        sys.exit(1)

    if args.browser:
        print(f"TransCrypting ejecutándose en {url} (Ctrl+C para cerrar)")
        webbrowser.open(url)
        try:
            while server_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    else:
        try:
            import webview

            initial_theme = "light"
            try:
                db.init_db()
                initial_theme = db.get_settings().get("theme", "light")
            except Exception:
                pass
            bg_color = "#181b1a" if initial_theme == "dark" else "#bcc5bd"

            bridge = DesktopBridge()
            window = webview.create_window(
                title="TransCrypting — Transcripción, Resumen y Chat Local",
                url=url,
                js_api=bridge,
                width=1360,
                height=860,
                min_size=(1024, 680),
                background_color=bg_color,
                text_select=True,
            )
            bridge.attach(window)
            window.events.shown += lambda: _apply_windows_titlebar_style(
                window, ICON_PATH, theme=initial_theme
            )
            webview.start(
                func=lambda: _apply_windows_titlebar_style(
                    window, ICON_PATH, theme=initial_theme
                ),
                icon=str(ICON_PATH) if ICON_PATH.exists() else None,
            )
        except Exception as exc:
            print(f"No se pudo abrir la ventana WebView2 ({exc}). Abriendo en navegador local...")
            _launch_edge_app_fallback(url)
            try:
                while server_thread.is_alive():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

    server.should_exit = True
    server_thread.join(timeout=8)


if __name__ == "__main__":
    main()
