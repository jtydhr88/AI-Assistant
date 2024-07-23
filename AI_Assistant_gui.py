from __future__ import annotations

import asyncio
import atexit
import os
import socket
import sys
import webbrowser
from threading import Event

from fastapi.responses import RedirectResponse
from fastapi import HTTPException
import base64
from PIL import Image
import io

from AI_Assistant_modules.actions.lighting import Lighting
from AI_Assistant_modules.application_config import ApplicationConfig
from AI_Assistant_modules.tab_gui import gradio_tab_gui
from modules import initialize
from modules import initialize_util
from modules import timer
from modules_forge import main_thread
from modules_forge.initialization import initialize_forge
from utils.img_utils import invert_process, multiply_images, mask_process, canny_process, \
    transparent_process, noline_process, positive_negative_shape_process
from utils.lang_util import LangUtil, get_language_argument

from AI_Assistant_modules.prompt_analysis import PromptAnalysis

from pydantic import BaseModel

from utils.request_api import prepare_image

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize_forge()
initialize.imports()
initialize.check_versions()
initialize.initialize()

shutdown_event = Event()

def create_api(app):
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(starting_port):
    port = starting_port
    while is_port_in_use(port):
        print(f"Port {port} is in use, trying next one.")
        port += 1
    return port


class ImageData(BaseModel):
    image_base64: str


class MultiplyImagesData(BaseModel):
    line_image_base64: str
    shadow_image_base64: str


class TransparentImagesData(BaseModel):
    image_base64: str
    transparent_threshold: int

class PositiveNegativeShapeImagesData(BaseModel):
    image_base64: str
    thresholds: list[int]
    colors: list[int]

class LightImageData(BaseModel):
    image_base64: str
    light_yaw: float
    light_pitch: float
    specular_power: float
    normal_diffuse_strength: float
    specular_highlights_strength: float
    total_gain: float


class CannyImageData(BaseModel):
    image_base64: str
    canny_threshold1: int
    canny_threshold2: int


async def api_only_worker(shutdown_event: Event):
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()

    # 言語設定の取得
    lang_util = LangUtil(get_language_argument())
    # 基準ディレクトリの取得
    if getattr(sys, 'frozen', False):
        # PyInstaller でビルドされた場合
        dpath = os.path.dirname(sys.executable)
    else:
        # 通常の Python スクリプトとして実行された場合
        dpath = os.path.dirname(sys.argv[0])
    app_config = ApplicationConfig(lang_util, dpath)

    #sys.argvの中に--exuiがある場合、app_configにexuiを設定する
    if "--exui" in sys.argv:
        app_config.exui = True

    # Gradioインターフェースの設定
    _, gradio_url, _ = gradio_tab_gui(app_config).queue().launch(share=False, prevent_thread_lock=True)

    # FastAPIのルートにGradioのURLへのリダイレクトを設定
    @app.get("/", response_class=RedirectResponse)
    async def read_root():
        return RedirectResponse(url=gradio_url)

    @app.post("/ai-assistant/prompt_analysis")
    async def api_prompt_analysis(data: ImageData):
        try:
            image_data = base64.b64decode(data.image_base64)
            image = Image.open(io.BytesIO(image_data))

            temp_path = 'temp_image.png'
            image.save(temp_path)

            prompt_analysis = PromptAnalysis(app_config=app_config, post_filter=False)

            tags_list = prompt_analysis.process_prompt_analysis(temp_path)

            import os
            os.remove(temp_path)

            return {"result": tags_list}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ai-assistant/mask_process")
    async def api_mask_process(data: ImageData):
        try:
            image_data = base64.b64decode(data.image_base64)
            image = Image.open(io.BytesIO(image_data))

            temp_path = 'temp_image.png'
            image.save(temp_path)

            mask_pil = mask_process(temp_path)

            import os
            os.remove(temp_path)

            return {"result": prepare_image(mask_pil)}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ai-assistant/canny_process")
    async def api_canny_process(data: CannyImageData):
        try:
            image_data = base64.b64decode(data.image_base64)
            image = Image.open(io.BytesIO(image_data))

            temp_path = 'temp_image.png'
            image.save(temp_path)

            canny_pil = canny_process(temp_path, data.canny_threshold1, data.canny_threshold2)

            import os
            os.remove(temp_path)

            return {"result": prepare_image(canny_pil)}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ai-assistant/invert_process")
    async def api_invert_process(data: ImageData):
        try:
            image_data = base64.b64decode(data.image_base64)
            image = Image.open(io.BytesIO(image_data))

            temp_path = 'temp_image.png'
            image.save(temp_path)

            invert_pil = invert_process(temp_path).resize(image.size, Image.LANCZOS).convert("RGB")

            import os
            os.remove(temp_path)

            return {"result": prepare_image(invert_pil)}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ai-assistant/light_process")
    async def api_light_process(data: LightImageData):
        try:
            image_data = base64.b64decode(data.image_base64)
            image = Image.open(io.BytesIO(image_data))

            temp_path = 'temp_image.png'
            image.save(temp_path)

            lighting = Lighting(app_config)

            light_pil = lighting._process(temp_path, data.light_yaw, data.light_pitch, data.specular_power,
                                          data.normal_diffuse_strength, data.specular_highlights_strength,
                                          data.total_gain)

            import os
            os.remove(temp_path)

            return {"result": prepare_image(light_pil)}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ai-assistant/multiply_images")
    async def api_multiply_images(data: MultiplyImagesData):
        try:
            line_image_data = base64.b64decode(data.line_image_base64)
            line_image = Image.open(io.BytesIO(line_image_data))

            shadow_image_data = base64.b64decode(data.shadow_image_base64)
            shadow_image = Image.open(io.BytesIO(shadow_image_data))

            result = multiply_images(line_image, shadow_image).convert("RGB")

            return {"result": prepare_image(result)}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ai-assistant/noline_process")
    async def api_noline_process(data: ImageData):
        try:
            image_data = base64.b64decode(data.image_base64)
            image = Image.open(io.BytesIO(image_data))

            temp_path = 'temp_image.png'
            image.save(temp_path)

            noline_pil = noline_process(temp_path).convert("RGB")

            import os
            os.remove(temp_path)

            return {"result": prepare_image(noline_pil)}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ai-assistant/transparent_process")
    async def api_transparent_process(data: TransparentImagesData):
        try:
            image_data = base64.b64decode(data.image_base64)
            image = Image.open(io.BytesIO(image_data))

            temp_path = 'temp_image.png'
            image.save(temp_path)

            transparent_pil = transparent_process(temp_path, data.transparent_threshold)

            import os
            os.remove(temp_path)

            return {"result": prepare_image(transparent_pil)}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ai-assistant/positive_negative_shape_process")
    async def api_positive_negative_shape_process(data: PositiveNegativeShapeImagesData):
        try:
            image_data = base64.b64decode(data.image_base64)
            image = Image.open(io.BytesIO(image_data))

            temp_path = 'temp_image.png'
            image.save(temp_path)

            thresholds = data.thresholds
            colors = data.colors

            if len(thresholds) < 3:
                raise HTTPException(status_code=500, detail=str("thresholds must have at least 3 elements"))

            if len(colors) < 1:
                raise HTTPException(status_code=500, detail=str("colors must have at least 3 elements"))

            if len(colors) != len(thresholds) - 2:
                raise HTTPException(status_code=500, detail=str("colors size must equip thresholds size - 2"))

            positive_negative_shape_pil = positive_negative_shape_process(temp_path, thresholds, colors)

            import os
            os.remove(temp_path)

            return {"result": prepare_image(positive_negative_shape_pil)}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    initialize_util.setup_middleware(app)
    api = create_api(app)

    from modules import script_callbacks
    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    print(f"Web UI is running at {gradio_url}.")
    webbrowser.open(gradio_url)

    starting_port = 7861
    port = find_available_port(starting_port)
    app_config.set_fastapi_url(f"http://127.0.0.1:{port}")

    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config=config)

    loop = asyncio.get_event_loop()
    shutdown_event.set()
    await loop.create_task(server.serve())

def api_only():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(api_only_worker(shutdown_event))

def on_exit():
    print("Cleaning up...")
    shutdown_event.set()

if __name__ == "__main__":
    atexit.register(on_exit)
    api_only()
    main_thread.loop()