import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, 'models/custom.joblib')
PREDICTOR_MODE = os.environ.get('PREDICTOR_MODE', 'heuristic').lower()

app = FastAPI()
app.mount('/static', StaticFiles(directory=os.path.join(APP_DIR, 'static')), name='static')
templates = Jinja2Templates(directory=os.path.join(APP_DIR, 'templates'))

# lazy load predictor
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        try:
            if PREDICTOR_MODE == 'heuristic':
                from infer import GreenPixelPredictor
                predictor = GreenPixelPredictor()
            else:
                from infer import Predictor
                if not os.path.exists(MODEL_PATH):
                    raise RuntimeError(f'Model not found at {MODEL_PATH}. Please train and save model.joblib')
                predictor = Predictor(MODEL_PATH)
        except ImportError as exc:
            raise RuntimeError('Unable to import inference utilities. Ensure the project root is on PYTHONPATH.') from exc
    return predictor


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/predict', response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    tmp_dir = os.path.join(APP_DIR, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    file_path = os.path.join(tmp_dir, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    error = None
    prediction = None

    from PIL import Image

    try:
        with Image.open(file_path) as image_file:
            img = image_file.convert('RGB')
        try:
            predictor_instance = get_predictor()
        except RuntimeError as exc:
            error = str(exc)
        else:
            try:
                prediction = predictor_instance.predict(img)
            except Exception as exc:
                error = f'Prediction failed: {exc}'
    except Exception as exc:
        error = f'Unable to read image: {exc}'
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

    context = {'request': request}
    if error:
        context['error'] = error
    else:
        context['prediction'] = prediction

    return templates.TemplateResponse('result.html', context)
