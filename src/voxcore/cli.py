"""CLI — Interface en ligne de commande VoxCore."""
import sys
import click
from pathlib import Path

from .config import load_config
from .pipeline import VoicePipeline
from .stt import STTEngine
from .llm import LLMEngine
from .tts import VoxtralTTS


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def main(ctx, config_path):
    """VoxCore — Local voice assistant pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@main.command()
@click.argument("audio", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output audio file path")
@click.pass_context
def live(ctx, audio, output):
    """Pipeline complet : audio → réponse vocale."""
    cfg = ctx.obj["config"]
    pipeline = VoicePipeline(cfg)
    
    click.echo(f"🎙  Transcription de {audio}...")
    response = pipeline.run(audio, output)
    
    click.echo(f"✉️  Réponse sauvegardée")
    click.echo(f"💬 {response}")


@main.command("stt")
@click.argument("audio", type=click.Path(exists=True))
@click.option("--language", "-l", default=None, help="Langue (fr, en, etc.)")
@click.pass_context
def stt(ctx, audio, language):
    """Transcrire un fichier audio en texte."""
    cfg = ctx.obj["config"]
    stt_engine = STTEngine(
        model_size=cfg["stt"]["model_size"],
        device=cfg["stt"]["device"],
        language=language or cfg["stt"]["language"],
    )
    
    text = stt_engine.transcribe(audio, language)
    click.echo(text)


@main.command("tts")
@click.argument("text")
@click.option("--output", "-o", default="output.mp3", help="Output audio file")
@click.option("--voice", "-v", default=None, help="Voice ID")
@click.option("--speed", "-s", default=1.0, type=float, help="Speed (0.5-2.0)")
@click.pass_context
def tts(ctx, text, output, voice, speed):
    """Synthétiser du texte en audio."""
    cfg = ctx.obj["config"]
    tts_engine = VoxtralTTS(
        url=cfg["tts"]["url"],
        voice=voice or cfg["tts"]["voice"],
        speed=speed,
        chunk_chars=cfg["tts"]["chunk_chars"],
    )
    
    path = tts_engine.synthesize_file(text, output, fmt=Path(output).suffix.lstrip("."))
    click.echo(f"🔊 Sauvegardé : {path}")


@main.command()
@click.pass_context
def status(ctx):
    """Vérifier l'état des services."""
    cfg = ctx.obj["config"]
    
    click.echo("=== VoxCore Status ===\n")
    
    # TTS health
    import urllib.request
    tts_url = cfg["tts"]["url"]
    health_url = tts_url.replace("/tts", "/health")
    try:
        with urllib.request.urlopen(health_url, timeout=5) as r:
            data = r.read().decode()
            click.echo(f"✅ TTS ({tts_url}) : {data}")
    except Exception as e:
        click.echo(f"❌ TTS ({tts_url}) : {e}")
    
    # LLM health
    llm_url = cfg["llm"]["base_url"].replace("/v1", "")
    try:
        with urllib.request.urlopen(f"{llm_url}/v1/models", timeout=5) as r:
            click.echo(f"✅ LLM ({llm_url}) : OK")
    except Exception as e:
        click.echo(f"❌ LLM ({llm_url}) : {e}")
    
    click.echo(f"\n📄 Config : {Path(ctx.obj.get('_config_path', '~/.config/voxcore/config.yaml'))}")


if __name__ == "__main__":
    main()
