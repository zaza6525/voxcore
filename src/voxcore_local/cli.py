"""CLI — Interface en ligne de commande VoxCore."""
import sys
import click
from pathlib import Path

from .config import load_config
from .pipeline import VoicePipeline
from .stt import STTEngine
from .tts import VoxtralTTS
from .live import LiveSession
from .tts_router import TTSRouter
from .microphone import MicrophoneCapture
from .llm import LLMEngine
from .memory_manager import MemoryManager


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def main(ctx, config_path):
    """VoxCore — Assistant vocal 100% local.
    
    STT → LLM → TTS, sans cloud, sans abonnement.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@main.command()
@click.pass_context
def live(ctx):
    """Mode live : micro → réponse vocale en continu.
    
    Parle, l'IA écoute et répond vocalement.
    Ctrl+C pour arrêter.
    """
    cfg = ctx.obj["config"]
    
    click.echo("🚀 Démarrage VoxCore LIVE...")
    
    session = LiveSession(cfg)
    session.start()


@main.command()
@click.argument("audio", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output audio file path")
@click.pass_context
def pipeline(ctx, audio, output):
    """Pipeline complet : fichier audio → réponse vocale."""
    cfg = ctx.obj["config"]
    pipe = VoicePipeline(cfg)
    
    click.echo(f"🎙  Transcription de {audio}...")
    response = pipe.run(audio, output)
    
    click.echo(f"💬 {response}")
    click.echo(f"✉️  Audio sauvegardé")


@main.command("stt")
@click.argument("audio", type=click.Path(exists=True))
@click.option("--language", "-l", default=None, help="Langue (fr, en, etc.)")
@click.pass_context
def stt(ctx, audio, language):
    """Transcrire un fichier audio en texte."""
    cfg = ctx.obj["config"]
    engine = STTEngine(
        model_size=cfg["stt"]["model_size"],
        device=cfg["stt"]["device"],
        language=language or cfg["stt"]["language"],
    )
    
    click.echo(engine.transcribe(audio, language))


@main.command("tts")
@click.argument("text")
@click.option("--output", "-o", default="output.mp3", help="Output audio file")
@click.option("--voice", "-v", default=None, help="Voice ID")
@click.option("--speed", "-s", default=1.0, type=float, help="Vitesse (0.5-2.0)")
@click.pass_context
def tts(ctx, text, output, voice, speed):
    """Synthétiser du texte en audio."""
    cfg = ctx.obj["config"]
    engine = VoxtralTTS(
        url=cfg["tts"]["url"],
        voice=voice or cfg["tts"]["voice"],
        speed=speed,
        chunk_chars=cfg["tts"]["chunk_chars"],
    )
    
    path = engine.synthesize_file(
        text, output, fmt=Path(output).suffix.lstrip(".") or "mp3"
    )
    click.echo(f"🔊 {path}")


@main.command()
@click.option("--model", "-m", default=None, help="Taille du modèle whisper")
@click.option("--device", "-d", default=None, help="cuda ou cpu")
@click.pass_context
def status(ctx, model, device):
    """Vérifier l'état des services et backends."""
    cfg = ctx.obj["config"]
    
    click.echo("=" * 50)
    click.echo("  VoxCore Status")
    click.echo("=" * 50)
    
    # TTS health
    router = TTSRouter(cfg)
    health = router.health_check()
    
    for backend, info in health.items():
        status_icon = "✅" if info["status"] == "ok" else "❌" if "error" in info["status"] else "⬜"
        click.echo(f"\n{status_icon} {backend.upper()}")
        for k, v in info.items():
            if k != "status":
                click.echo(f"   {k}: {v}")
    
    # LLM health
    llm_url = cfg["llm"]["base_url"].replace("/v1", "")
    import urllib.request
    try:
        with urllib.request.urlopen(f"{llm_url}/v1/models", timeout=5) as r:
            data = r.read().decode()
            click.echo(f"\n✅ LLM ({llm_url})")
            click.echo(f"   models: {data[:100]}...")
    except Exception as e:
        click.echo(f"\n❌ LLM ({llm_url})")
        click.echo(f"   error: {e}")
    
    # STT model
    click.echo(f"\n🎙  STT")
    click.echo(f"   model: {model or cfg['stt']['model_size']}")
    click.echo(f"   device: {device or cfg['stt']['device']}")
    click.echo(f"   lang: {cfg['stt']['language']}")
    
    click.echo(f"\n📄 Config: {cfg}")


@main.command()
@click.pass_context
def devices(ctx):
    """Lister les devices audio disponibles."""
    mic = MicrophoneCapture()
    devices = mic.list_devices()
    
    click.echo("Microphones disponibles :")
    for d in devices:
        click.echo(f"  [{d['index']}] {d['name']} ({d['channels']}ch, {d['rate']}Hz)")


@main.command()
@click.pass_context
def chat(ctx):
   """Mode chat CLI avec mémoire persistante (texte → texte via LLM local)."""
   cfg = ctx.obj["config"]
   llm = LLMEngine(
       base_url=cfg["llm"]["base_url"],
       model=cfg["llm"]["model"],
       api_key=cfg["llm"]["api_key"],
   )
    
   memory = MemoryManager(cfg.get("memory", {}))
   history = [{"role": "system", "content": cfg["llm"]["system_prompt"]}]
    
   click.echo("💬 VoxCore Chat (Ctrl+C pour quitter)")
   click.echo("   /stats → statistiques mémoire")
   click.echo("   /clear → nouvelle session")
   click.echo("-" * 40)
    
   while True:
       try:
           user = click.prompt("")
       except (KeyboardInterrupt, EOFError):
           click.echo("\nAu revoir !")
           break
        
       if user == "/stats":
           stats = memory.get_stats()
           for k, v in stats.items():
               click.echo(f"  {k}: {v}")
           click.echo("-" * 40)
           continue
        
       if user == "/clear":
           memory.new_session()
           click.echo("Session réinitialisée (mémoire long-terme conservée)")
           click.echo("-" * 40)
           continue
        
       history.append({"role": "user", "content": user})
        
       # Store in memory
       memory.process_event(content=user, category="conversation", source="user")
        
       # Build context with memory
       context = memory.build_context(history, top_k=5)
       messages = memory.inject_context(history, context)
        
       response = llm.chat(messages, max_tokens=cfg["llm"]["max_tokens"])
       history.append({"role": "assistant", "content": response})
        
       # Store response in memory
       memory.process_event(content=response, category="conversation", source="assistant")
        
       click.echo(response)
       click.echo("-" * 40)
        
       # Garne le contexte
       while len(history) > 21:
           history.pop(1)


if __name__ == "__main__":
    main()
