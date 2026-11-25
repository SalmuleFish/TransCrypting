# TransCrypting — Transcripción, Resumen, Chat y Modo Examen 100% Local

Aplicación de escritorio para Windows diseñada para trabajar **exclusivamente en tu computadora**, sin enviar ningún archivo de audio, transcripción ni conversación a la nube. Ideal para grabar o subir clases, obtener resúmenes estructurados, hacer preguntas con citas al minuto exacto y repasar con exámenes interactivos.

---

## Capturas de pantalla

### Vista de resumen e interfaz Bento (Modo Claro)
Resumen estructurado automático con puntos clave, acuerdos para el examen, marcas de tiempo interactivas `▶ MM:SS` y telemetría de hardware en tiempo real.

![Vista de resumen en modo claro](docs/screenshots/summary-light.png)

### Chat contextual y citas al segundo exacto (Modo Oscuro Grafito)
Preguntas y respuestas basadas en la clase con citas interactivas que saltan directo al audio, sugerencias dinámicas y medidor de salud de contexto (`Óptimo` / `RAG Protegido`).

![Chat contextual con citas interactivas en modo oscuro](docs/screenshots/chat-dark.png)

### Simulador de examen interactivo
Preguntas de opción múltiple con retroalimentación instantánea y preguntas de respuesta abierta evaluadas por tu modelo local frente a lo explicado en clase.

![Modo examen interactivo con corrección local](docs/screenshots/exam-dark.png)

### Modelos modulares y guía integrada
Compatibilidad flexible con cualquier modelo de Ollama o Whisper / CTranslate2 desde la ventana de **Configuración**.

![Guía integrada para instalar modelos modulares](docs/screenshots/models-guide.png)

---

## Características principales

- **Transcripción local con Whisper (`faster-whisper`)**: aceleración en GPU NVIDIA (`CUDA int8_float16`) con respaldo automático en CPU, detección bilingüe por segmento (español/inglés) y preprocesamiento opcional con `ffmpeg` para clases con ruido o voces superpuestas.
- **Modelos de lenguaje modulares vía Ollama**: detecta automáticamente cualquier modelo instalado en tu Ollama local (`Gemma 4`, `Qwen 3 / 3.5`, `Llama 3.2`, etc.) sin depender de un único modelo fijo.
- **Múltiples grabaciones por conversación + Protección anti-saturación**: adjunta más audios a una misma sesión para generar un **Resumen integrado**. Cuando el contexto supera el límite seguro de `num_ctx`, activa automáticamente el modo **RAG Protegido (BM25 local)** para evitar saturar el modelo.
- **Citas interactivas `▶ MM:SS`**: cada marca de tiempo en el Chat, Resumen, Transcripción o Examen es un botón que salta y reproduce el audio en ese segundo exacto.
- **Modo Examen interactivo**: genera preguntas de **opción múltiple** (corrección al instante) y **respuesta abierta** (evaluada por el modelo local frente a lo explicado en clase, con enlace al minuto exacto).
- **Gestión inteligente de VRAM y RAM**: Whisper y el modelo de Ollama se turnan de forma exclusiva y la memoria se libera automáticamente tras inactividad (o al instante con **Liberar VRAM**).
- **Interfaz Bento con Modo Claro y Modo Oscuro Grafito**, exportación nativa a `.md` y opción **Borrar historial** de chat conservando intactas grabaciones, transcripciones y resúmenes.

---

## Requisitos

- **Windows 10 / 11** (64-bit)
- **Python 3.11 o 3.12**
- **Ollama** instalado localmente ([ollama.com](https://ollama.com)) con al menos un modelo descargado (por ejemplo `ollama pull gemma4:e4b` o `ollama pull qwen3:8b`)
- **ffmpeg** disponible en el `PATH` del sistema
- *(Opcional)* GPU NVIDIA compatible con CUDA para acelerar Whisper y Ollama

---

## Instalación rápida (al clonar desde GitHub)

1. Clona el repositorio y entra en la carpeta:
   ```powershell
   git clone https://github.com/SalmuleFish/TransCrypting.git
   cd TransCrypting
   ```
2. Ejecuta **`setup.bat`** (doble clic o desde terminal). El script creará `.venv`, instalará las dependencias de `requirements.txt` y compilará **`TransCrypting.exe`** con su icono `TC`.
3. Descarga tu primer modelo local en Ollama si aún no tienes uno:
   ```powershell
   ollama pull gemma4:e4b
   ```

---

## Cómo usar el programa

### Abrir la aplicación
- Haz doble clic en **`TransCrypting.exe`** (puedes crear un acceso directo en el Escritorio o anclarlo a la barra de tareas).
- O desde terminal:
  - Ventana de escritorio: `.\.venv\Scripts\python.exe run.py`
  - Navegador local: `.\.venv\Scripts\python.exe run.py --browser`
  - Solo API local: `.\.venv\Scripts\python.exe run.py --headless`

### Modelos modulares (instalar los que prefieras)
Dentro de la app abre **Configuración** (icono de engrane) y pulsa **`¿Cómo instalar modelos?`** para ver la guía interactiva y copiar comandos con un clic:
- **Ollama (Resumen / Chat / Examen)**: descarga cualquier modelo con `ollama pull <modelo>` (ej. `llama3.2:3b`, `gemma4:e4b`, `qwen3:8b`, `qwen3.5:9b`) y pulsa **Actualizar lista** en Configuración.
- **Whisper (Transcripción)**: elige entre `tiny`, `base`, `small`, `medium`, `large-v3-turbo` y `large-v3`, o coloca cualquier modelo personalizado en formato **CTranslate2** (`model.bin` + `config.json`) dentro de `data/models/whisper/<tu-modelo>/`.

---

## Estructura del proyecto y privacidad en Git

```text
TransCrypting/
├── backend/                # API FastAPI, gestión exclusiva de VRAM, Whisper, Ollama y RAG BM25
├── docs/screenshots/       # Capturas de pantalla incluidas en este README
├── frontend/               # Interfaz Bento 100% offline (HTML/CSS/JS + icon.ico)
├── launcher/               # Código fuente C# para compilar TransCrypting.exe
├── tests/                  # Pruebas automatizadas con pytest
├── data/                   # SOLO LOCAL (ignorado en Git; únicamente se versiona .gitkeep)
│   ├── audio/              # Grabaciones subidas por el usuario
│   ├── models/             # Pesos cacheados/personalizados de Whisper
│   └── transcript.db       # Conversaciones, transcripciones, resúmenes y chat
├── .gitignore              # Protege data/, .venv/, cachés y compilados locales
├── requirements.txt
├── setup.bat               # Instalador en Windows
└── run.py
```

Ningún archivo en `data/`, `.venv/`, `__pycache__/`, `.pytest_cache/` o `TransCrypting.exe` se sube a GitHub gracias a `.gitignore`.
