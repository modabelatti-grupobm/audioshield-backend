"""
AudioShield Backend
===================
Dual-layer audio injection API.
- Recebe um vídeo (MP4/MOV)
- Injeta um sinal "fantasma" nas frequências 4-8 kHz que STT prioriza
- Devolve o vídeo com áudio humano intacto + camada fantasma embutida

Deploy: Railway / Render / Fly.io
"""

import os
import uuid
import tempfile
import subprocess
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import soundfile as sf
from scipy import signal

app = Flask(__name__)
CORS(app)  # permite chamadas do frontend Lovable

MAX_FILE_MB = 500

# ── Dual-layer injection ──────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 44100):
    """Extrai áudio do vídeo via ffmpeg, retorna (samples float32, sr)."""
    wav_path = path + "_extracted.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", path,
        "-vn", "-acodec", "pcm_f32le",
        "-ar", str(target_sr), "-ac", "1",
        wav_path
    ], check=True, capture_output=True)
    data, sr = sf.read(wav_path)
    os.unlink(wav_path)
    return data.astype(np.float32), sr


def inject_ghost_layer(y: np.ndarray, sr: int, preset: str = "MAX") -> np.ndarray:
    """
    Injeta camada fantasma: sinal de ruído estruturado nas bandas
    4–8 kHz que modelos STT (Whisper, Google, Azure) priorizam para
    detecção de fonemas. O ouvido humano tem baixa sensibilidade nessa
    faixa em amplitudes abaixo de ~1% do sinal principal.

    Técnicas combinadas:
    1. Ruído adversarial de alta frequência (band-limited noise)
    2. Chirp sweeps periódicos (confunde alinhamento temporal do STT)
    3. Jitter de fase espectral por janela STFT
    4. Eco fantasma sub-perceptível (<20ms)
    5. Micro-variação de pitch por segmento
    """

    configs = {
        "MAX":  dict(noise_amp=0.042, chirp_amp=0.018, phase_str=0.72, echo_ms=11, echo_g=0.11, pitch_st=0.17, pitch_ms=45,  stretch=0.97, gain=0.82),
        "HIGH": dict(noise_amp=0.026, chirp_amp=0.010, phase_str=0.52, echo_ms=14, echo_g=0.07, pitch_st=0.11, pitch_ms=75,  stretch=0.98, gain=0.88),
        "MED":  dict(noise_amp=0.013, chirp_amp=0.005, phase_str=0.28, echo_ms=18, echo_g=0.04, pitch_st=0.06, pitch_ms=110, stretch=0.99, gain=0.92),
    }
    cfg = configs.get(preset, configs["MAX"])

    # 1. Band-limited adversarial noise (4–8 kHz)
    noise = np.random.randn(len(y)).astype(np.float32) * cfg["noise_amp"]
    nyq = sr / 2.0
    b, a = signal.butter(4, [4000/nyq, min(8000/nyq, 0.99)], btype="band")
    noise = signal.filtfilt(b, a, noise).astype(np.float32)

    # 2. Chirp sweeps (100ms cada, varre 4k→8k Hz)
    chirp_layer = np.zeros(len(y), dtype=np.float32)
    chirp_len = int(sr * 0.1)
    for start in range(0, len(y) - chirp_len, int(sr * 0.3)):
        t = np.linspace(0, 0.1, chirp_len)
        c = signal.chirp(t, f0=4000, f1=8000, t1=0.1, method="linear").astype(np.float32)
        chirp_layer[start:start+chirp_len] += c * cfg["chirp_amp"]

    # 3. Phase jitter via STFT
    n_fft = 2048
    hop = 512
    win = "hann"
    f, t_stft, Zxx = signal.stft(y, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft-hop)
    mag   = np.abs(Zxx)
    phase = np.angle(Zxx)
    mask  = np.random.rand(*phase.shape) < cfg["phase_str"]
    rand_phase = np.random.uniform(-np.pi, np.pi, phase.shape)
    phase_jittered = np.where(mask, rand_phase, phase)
    Zxx_new = mag * np.exp(1j * phase_jittered)
    _, y_phase = signal.istft(Zxx_new, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft-hop)
    y_phase = y_phase[:len(y)].astype(np.float32)

    # 4. Ghost echo (<20ms — below auditory fusion threshold)
    delay_samples = int(sr * cfg["echo_ms"] / 1000)
    echo = np.zeros(len(y), dtype=np.float32)
    echo[delay_samples:] = y[:len(y)-delay_samples] * cfg["echo_g"]

    # 5. Pitch micro-variation per segment
    seg_len = int(sr * cfg["pitch_ms"] / 1000)
    y_pitch = np.zeros(len(y), dtype=np.float32)
    for i in range(0, len(y), seg_len):
        chunk = y[i:i+seg_len]
        semitones = np.random.uniform(-cfg["pitch_st"], cfg["pitch_st"])
        ratio = 2 ** (semitones / 12)
        new_len = len(chunk)
        src_indices = np.arange(new_len) * ratio
        src_indices = np.clip(src_indices, 0, len(chunk) - 1.001)
        idx = src_indices.astype(int)
        frac = src_indices - idx
        shifted = chunk[idx] + frac * (chunk[np.clip(idx+1, 0, len(chunk)-1)] - chunk[idx])
        y_pitch[i:i+len(shifted)] = shifted.astype(np.float32)

    # 6. Time-stretch alternado por bloco de 1s
    block = sr
    y_stretch = np.zeros(len(y), dtype=np.float32)
    for bi, start in enumerate(range(0, len(y), block)):
        chunk = y[start:start+block]
        f_rate = cfg["stretch"] if bi % 2 == 0 else 1.0 / cfg["stretch"]
        new_len = int(len(chunk) * f_rate)
        if new_len < 2:
            continue
        src = (np.arange(len(chunk)) / f_rate).astype(np.float32)
        src = np.clip(src, 0, len(chunk) - 1.001)
        idx = src.astype(int)
        frac = src - idx
        stretched = chunk[idx] + frac * (chunk[np.clip(idx+1, 0, len(chunk)-1)] - chunk[idx])
        end = min(start + len(chunk), len(y_stretch))
        copy_len = min(len(stretched), end - start)
        y_stretch[start:start+copy_len] = stretched[:copy_len]

    # Combine: original + all ghost layers
    ghost = noise + chirp_layer + echo
    # Phase-jitter and pitch layers blend at low amplitude
    ghost += y_phase * 0.04 + y_pitch * 0.03 + y_stretch * 0.02

    # Mix: keep original dominant, ghost below perceptibility
    combined = y + ghost

    # Normalize
    peak = np.max(np.abs(combined))
    if peak > 0:
        combined = combined / peak * cfg["gain"]

    return combined.astype(np.float32)


def mux_audio_to_video(video_path: str, audio: np.ndarray, sr: int, output_path: str):
    """Salva áudio processado e remuxa com o vídeo original (sem recodificar vídeo)."""
    wav_tmp = video_path + "_processed.wav"
    sf.write(wav_tmp, audio, sr, subtype="FLOAT")

    ext = os.path.splitext(video_path)[1].lower()
    codec = "aac"
    extra = ["-b:a", "192k"]

    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", wav_tmp,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", codec,
        *extra,
        "-shortest",
        output_path
    ], check=True, capture_output=True)

    os.unlink(wav_tmp)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    f = request.files["file"]
    preset = request.form.get("preset", "MAX").upper()
    if preset not in ("MAX", "HIGH", "MED"):
        preset = "MAX"

    # Check size
    f.seek(0, 2)
    size_mb = f.tell() / 1024 / 1024
    f.seek(0)
    if size_mb > MAX_FILE_MB:
        return jsonify({"error": f"Arquivo muito grande. Máximo: {MAX_FILE_MB} MB"}), 413

    ext = os.path.splitext(f.filename)[1].lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
        return jsonify({"error": "Formato não suportado. Use MP4 ou MOV."}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, f"input{ext}")
        output_path = os.path.join(tmpdir, f"output{ext}")

        f.save(input_path)

        try:
            # Extract + process audio
            y, sr = load_audio(input_path)
            y_protected = inject_ghost_layer(y, sr, preset)
            mux_audio_to_video(input_path, y_protected, sr, output_path)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": "Erro ao processar vídeo: " + (e.stderr.decode() if e.stderr else str(e))}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        mime = "video/quicktime" if ext == ".mov" else "video/mp4"
        base = os.path.splitext(f.filename)[0]
        download_name = f"{base}_protegido{ext}"

        return send_file(
            output_path,
            mimetype=mime,
            as_attachment=True,
            download_name=download_name,
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
