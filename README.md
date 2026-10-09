# Vidence — MVP

Gestión local de evidencia multimedia personal: **captura → mejora con IA local → transcripción → resguardo cifrado**, 100 % offline.

Vidence es una herramienta neutral: no evalúa, juzga ni clasifica el contenido. Solo captura, mejora la calidad, transcribe y resguarda.

## Qué demuestra este MVP

- **IA local 1, limpieza de audio:** RNNoise (red neuronal, filtro `arnndn` de ffmpeg) más normalización EBU R128 en dos pasadas.
- **IA local 2, transcripción:** faster-whisper en CPU con cuantización `int8`, VAD Silero y marcas de tiempo por palabra.
- **IA local 3, análisis visual de video:** YOLOv8 Nano en ONNX con OpenCV DNN en CPU; detección fáctica de personas y objetos en fotogramas clave con cajas delimitadoras y etiquetas.
- **Video:** reducción de ruido y nitidez con ffmpeg, más fotogramas clave con OpenCV (uno cada N segundos).
- **Resguardo:** SHA-256 del original al recibirlo; todo se guarda cifrado con AES-256-GCM; el original nunca se modifica.
- **Interfaz:** grabación desde el navegador o subida de archivos; comparación A/B entre original y mejorado; transcripción sincronizada; galería de fotogramas con cajas delimitadoras y filtros de objetos; reporte JSON.

## Cómo se cumplen las restricciones

| Requisito | Implementación | Verificado por |
|---|---|---|
| Cero salida de datos | `app/offline_guard.py` bloquea `connect`, DNS, `sendto` y `asyncio.sock_connect` hacia todo destino que no sea loopback. Se activa al importar `app`. | `tests/test_offline_guard.py` (sockets reales, sync y asyncio) |
| Modelos solo locales | `HF_HUB_OFFLINE=1`, carga desde ruta local con `local_files_only=True` y verificación SHA-256 contra `models/MANIFEST.json` | `tests/test_model_files.py` |
| Sin telemetría ni SDKs de nube | Ninguna dependencia directa de nube; sin extras de FastAPI (traen exportadores OTLP); `OTEL_SDK_DISABLED` | `tests/test_supply_chain.py` |
| Navegador sin salida | CSP `default-src 'self'; connect-src 'self'`, sin CDN ni fuentes externas. `/docs` está desactivado porque carga JS de un CDN. | `tests/test_api.py` |
| ffmpeg sin red | `-protocol_whitelist file` y `-format_whitelist` en cada llamada (bloquea playlists HLS maliciosas) | `tests/test_media.py` |
| Cifrado AES-256 en reposo | `app/crypto.py`: AES-256-GCM por bloques (STREAM) con DEK por archivo y KEK derivada con Argon2id (256 MiB) | `tests/test_crypto.py` (alteración, truncado, reordenamiento, trasplante) |
| Original inalterable | Hash al recibir; se verifica antes de procesar; cada derivado registra `derived_from_sha256` y los argumentos exactos de ffmpeg | `tests/test_pipeline.py` |
| Sin clasificación de riesgo | Los esquemas (`app/schemas.py`) no tienen campos de riesgo, amenaza ni puntuación | revisión de esquema |
| Hardware limitado | Whisper en CPU `int8`; un solo worker procesa en orden | — |

## Requisitos

- Windows 10/11 (probado en Windows 11), Python **3.14.7** (probado), ffmpeg 7 o posterior (probado con 9.0.2).
- Unos 600 MB de disco para los modelos (Whisper `small` + `tiny` y RNNoise).

## Instalación

```powershell
winget install --id Gyan.FFmpeg -e          # si aún no tiene ffmpeg
cd backend
uv venv .venv --python 3.14
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# ÚNICO paso con internet: descarga los modelos y escribe el manifiesto SHA-256.
.venv\Scripts\python.exe scripts\provision_models.py

copy .env.example .env   # ajuste VIDENCE_FFMPEG_BIN/FFPROBE_BIN si ffmpeg no está en el PATH
```

## Uso

```powershell
cd backend
.venv\Scripts\python.exe -m app
```

En el primer arranque se le pedirá crear la frase de acceso de la bóveda (mínimo 10 caracteres). Después abra **http://127.0.0.1:8000**.

> **Si pierde la frase de acceso, la evidencia no se puede recuperar.** Anótela en papel y guárdela en un lugar seguro.

## Demo guiada (5 minutos)

1. **Opcional:** desconecte el WiFi o el cable de red para demostrar que todo funciona offline.
2. Abra http://127.0.0.1:8000. En la cabecera deben aparecer: *Offline · red bloqueada*, *Whisper small*, *CPU int8* y *RNNoise*.
3. Pestaña **Subir archivo** → `samples/demo_voz_con_ruido.wav` (voz sintética en inglés con ruido rosa añadido). Para probar video use `samples/demo_video_con_ruido.mp4`.
4. Siga las etapas: recepción y cifrado → verificación → metadatos → mejora → transcripción → resguardo.
5. En el resultado:
   - **Alternar A/B** cambia entre original y mejorado en el mismo instante, para escuchar el ruido eliminado.
   - Haga clic en una **marca de tiempo** para saltar a ese momento del audio.
   - Las **palabras de baja confianza** aparecen subrayadas.
   - Revise **Metadatos e integridad** y **Cadena de custodia**.
6. Pestaña **Grabar**: grabe audio con el micrófono o video con la cámara. El video genera además la galería de fotogramas.
7. Descargue el **Reporte JSON**.

Medido en este equipo con Whisper `small` en CPU `int8`: un audio de 17 s se procesa completo en unos 7 s, y un video de 17 s en unos 9 s.

## Arquitectura

```
navegador ──stream──▶ POST /api/evidence ──▶ SHA-256 + AES-256-GCM (1 pasada) ──▶ data/vault/evidence/<id>/original.bin.enc
                                                                     │
worker (1 hilo):  descifrar a workspace efímero ─▶ verificar SHA-256 ─▶ ffprobe (metadatos)
   ├─ ffmpeg: vista previa sin filtros + versión mejorada (highpass → RNNoise → loudnorm → 48 kHz)
   ├─ ffmpeg → WAV 16 kHz mono ─▶ faster-whisper (CPU int8, VAD, marcas por palabra)
   └─ OpenCV: 1 fotograma cada N s del video mejorado (JPEG en memoria → cifrado directo)
   ─▶ cada derivado cifrado ─▶ manifest.json cifrado ─▶ borrar workspace
```

| Módulo (`backend/app/`) | Responsabilidad |
|---|---|
| `offline_guard.py` | Bloqueo de red del proceso (se instala al importar `app`) |
| `crypto.py`, `vault.py` | Formato cifrado por bloques, Argon2id, bóveda con nombres seguros y escritura atómica |
| `ingest.py` | Recepción en streaming: hash y cifrado sin escribir en claro |
| `media.py`, `frames.py`, `transcribe.py` | ffmpeg/ffprobe, OpenCV y Whisper |
| `pipeline.py`, `jobs.py` | Orquestación por etapas y cola de un worker |
| `api.py`, `main.py` | Rutas HTTP, Range, cabeceras de seguridad, CSRF, Host de confianza |
| `../frontend/` | HTML, CSS y JS sin dependencias ni recursos externos |

### Formato de archivo cifrado (`.enc`)

| Campo | Bytes | Contenido |
|---|---|---|
| magic | 4 | `VDNC` |
| versión | 1 | `0x01` |
| chunk_size | 4 | tamaño del bloque en claro (1 MiB) |
| nonce_prefix | 7 | aleatorio por archivo |
| wrapped_dek | 40 | clave del archivo envuelta con la KEK (AES Key Wrap) |
| bloques | n × (chunk + 16) | AES-256-GCM; nonce = prefijo ‖ contador ‖ bandera de último bloque; AAD = cabecera ‖ `vidence/<id>/<nombre>` |

Permite descifrar por rangos (*seek* en el reproductor) sin descifrar el archivo completo. Como la ubicación del objeto forma parte de la AAD, un `.enc` copiado sobre otro (o movido a otra evidencia) no se descifra.

## Pruebas

```powershell
cd backend
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest --cov=app
.venv\Scripts\python.exe -m ruff check app tests scripts
```

273 tests, 96 % de cobertura; `ruff` sin hallazgos. Los marcadores `ffmpeg` y `slow` identifican los tests que usan ffmpeg real y Whisper `tiny` real. Los tests usan solo medios sintéticos (lavfi y voz SAPI), nunca evidencia real.

### Prueba de estrés de punta a punta

```powershell
cd backend
.venv\Scripts\python.exe scripts\generate_test_corpus.py      # 18 videos/audios sintéticos -> samples\corpus\
.venv\Scripts\python.exe scripts\e2e_stress.py --iterations 20
```

El corpus cubre:

- video de teléfono vertical con frame rate variable;
- HEVC en MOV;
- WebM como el de MediaRecorder (sin duración en la cabecera);
- dimensiones impares y video sin audio;
- pista de silencio y audio telefónico de 8 kHz;
- MP3 con carátula y Ogg Vorbis;
- ruido intenso, un video de 90 s y un archivo con dos pistas de audio;
- un MP4 truncado, texto disfrazado, una trampa HLS y un ejecutable.

El script arranca un servidor real con una bóveda temporal desechable. Verifica en cada archivo:

- hashes y linaje de los derivados;
- descargas por rango;
- duración y orientación;
- ausencia de NaN y picos;
- número de fotogramas;
- transcripción (WER contra el texto pronunciado);
- los rechazos HTTP.

Al terminar busca texto en claro en la bóveda y escribe el reporte en `reports/`.
