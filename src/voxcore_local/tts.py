1|"""TTS — Text-to-Speech with multiple backends."""
2|import io
3|import json
4|import os
5|import re
6|import shutil
7|import subprocess
8|import sys
9|import tempfile
10|import urllib.request
11|import wave
12|from pathlib import Path
13|from typing import Optional
14|
15|import re as regex
16|
17|# Regex pour strip les emojis qui plantent certains TTS
18|_EMOJI_STRIP = re.compile(
19|    "["
20|    "\U0001F000-\U0001FFFF"
21|    "\u2600-\u27BF"
22|    "\uFE00-\uFE0F"
23|    "\u200B-\u200F"
24|    "\u202A-\u202E"
25|    "\u2060-\u206F"
26|    "\uFEFF"
27|    "]+"
28|)
29|
30|
31|def sanitize(text: str) -> str:
32|    """Nettoie le texte pour le TTS."""
33|    text = _EMOJI_STRIP.sub("", text)
34|    text = re.sub(r"\s+", " ", text)
35|    return text.strip()
36|
37|
38|def split_text(text: str, max_chars: int) -> list[str]:
39|    """Split text en chunks respectant les frontières de phrases."""
40|    max_chars = max(80, max_chars)
41|    sentence_parts = re.split(r"(?<=[.!?;:])\s*", text)
42|    chunks = []
43|    current = ""
44|
45|    for part in sentence_parts:
46|        part = part.strip()
47|        if not part:
48|            continue
49|
50|        if not current:
51|            current = part
52|            continue
53|
54|        if len(current) + 1 + len(part) <= max_chars:
55|            current = f"{current} {part}"
56|            continue
57|
58|        chunks.append(current)
59|        current = part
60|
61|    if current:
62|        chunks.append(current)
63|    return chunks or [text]
64|
65|
66|class VoxtralTTS:
67|    """TTS via Voxtral distant (PC2 style)."""
68|    
69|    def __init__(self, url: Optional[str] = None, 
70|                 voice: str = "fr_female", speed: float = 1.0,
71|                 chunk_chars: int = 320, timeout: float = 240.0):
72|        self.url = url or os.environ.get("VOXCORE_TTS_URL", "http://127.0.0.1:9000/tts")
73|        self.voice = voice
        self.speed = max(0.5, min(2.0, speed))
74|        self.chunk_chars = chunk_chars
75|        self.timeout = timeout
76|    
77|    def synthesize(self, text: str) -> bytes:
78|        """Synthétise du texte en WAV bytes."""
79|        text = sanitize(text)
80|        if not text:
81|            raise ValueError("Texte vide après nettoyage")
82|        
83|        chunks = split_text(text, self.chunk_chars)
84|        wav_parts = []
85|        
86|        for chunk in chunks:
87|            payload = json.dumps({
88|                "text": chunk,
89|                "voice": self.voice,
90|                "speed": self.speed,
91|            }).encode("utf-8")
92|            
93|            req = urllib.request.Request(
94|                self.url,
95|                data=payload,
96|                headers={"Content-Type": "application/json", "Accept": "audio/wav"},
97|                method="POST",
98|            )
99|            
100|            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
101|                wav_parts.append(resp.read())
102|        
103|        return join_wavs(wav_parts)
104|    
105|    def synthesize_file(self, text: str, output_path: str, fmt: str = "mp3") -> str:
106|        """Synthétise et écrit dans un fichier."""
107|        wav = self.synthesize(text)
108|        out = Path(output_path)
109|        out.parent.mkdir(parents=True, exist_ok=True)
110|        
111|        ext = out.suffix.lower().lstrip(".") or "mp3"
112|        if ext == "wav":
113|            out.write_bytes(wav)
114|        else:
115|            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
116|                tmp.write(wav)
117|                tmp_path = tmp.name
118|            
119|            ffmpeg = shutil.which("ffmpeg")
120|            if not ffmpeg:
121|                raise RuntimeError("ffmpeg requis pour la conversion")
122|            
123|            args = [ffmpeg, "-y", "-loglevel", "error", "-i", tmp_path]
124|            if ext == "mp3":
125|                args += ["-ac", "1", "-ar", "24000", "-b:a", "128k"]
126|            elif ext == "ogg":
127|                args += ["-acodec", "libopus", "-ac", "1", "-b:a", "64k"]
128|            args.append(str(out))
129|            
130|            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
131|            os.unlink(tmp_path)
132|        
133|        return str(out)
134|
135|
136|class LocalTTS:
137|    """TTS local via Piper (pour le fallback)."""
138|    
139|    def __init__(self, model_path: str = None, voice: str = "fr_google_feminine"):
140|        self.model_path = model_path or os.environ.get("PIPER_MODEL")
141|        self.voice = voice
142|    
143|    def synthesize(self, text: str) -> bytes:
144|        text = sanitize(text)
145|        cmd = ["piper", "--sentence_silence", "0.1"]
146|        if self.model_path:
147|            cmd += ["--model", self.model_path]
148|        cmd += ["--output-raw", "-"]
149|        
150|        proc = subprocess.run(cmd, input=text, capture_output=True, text=True, check=True)
151|        return proc.stdout
152|
153|
154|def join_wavs(wavs: list[bytes], silence_s: float = 0.0) -> bytes:
155|    """Joint plusieurs chunks WAV."""
156|    if len(wavs) == 1:
157|        return wavs[0]
158|    
159|    out = io.BytesIO()
160|    with wave.open(out, "wb") as dst:
161|        for i, wav_data in enumerate(wavs):
162|            with wave.open(io.BytesIO(wav_data), "rb") as src:
163|                params = src.getparams()
164|                if i == 0:
165|                    dst.setparams(params)
166|                if i > 0 and silence_s > 0:
167|                    silence = b"\x00" * int(params.framerate * silence_s) * params.nchannels * params.sampwidth
168|                    dst.writeframes(silence)
169|                dst.writeframes(src.readframes(params.nframes))
170|    
171|    return out.getvalue()
172|