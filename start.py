"""
start.py — Inicia o servidor Flask + túnel ngrok
=================================================
Rode com:
    python start.py

O script:
  1. Inicia o Flask na porta 8000 em uma thread separada
  2. Abre um túnel ngrok apontando para localhost:8000
  3. Imprime a URL pública compartilhável

Caso não tenha token ngrok configurado:
  - O Flask rodará normalmente em http://localhost:8000
  - Para obter a URL pública, acesse https://ngrok.com e crie uma conta gratuita,
    depois execute: ngrok config add-authtoken SEU_TOKEN
"""

import sys
import threading
import time
import webbrowser

# ---------------------------------------------------------------------------
# Start Flask in a background thread
# ---------------------------------------------------------------------------
def _run_flask():
    from app import app
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)


def main():
    print("\n" + "=" * 58)
    print("  IFRS 9 — Curso Interativo de Risco de Crédito")
    print("=" * 58)

    # Launch Flask
    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()
    time.sleep(2)  # Wait for Flask to be ready
    print(f"\n  ✓ Servidor Flask rodando em http://localhost:8000")

    # ---------------------------------------------------------------------------
    # Try ngrok
    # ---------------------------------------------------------------------------
    try:
        from pyngrok import ngrok, conf

        # Optional: set custom ngrok path if needed
        # conf.get_default().ngrok_path = "/path/to/ngrok"

        public_url = ngrok.connect(8000, "http")
        url_str = str(public_url)

        print(f"  ✓ Túnel ngrok ativo")
        print(f"\n{'='*58}")
        print(f"  URL LOCAL  : http://localhost:8000")
        print(f"  URL PÚBLICA: {url_str}")
        print(f"{'='*58}")
        print("\n  Compartilhe a URL pública com qualquer pessoa!")
        print("  Pressione Ctrl+C para encerrar.\n")

        # Abrir no browser automaticamente
        try:
            webbrowser.open("http://localhost:8000")
        except Exception:
            pass

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Encerrando servidor...")
            ngrok.disconnect(public_url)
            ngrok.kill()

    except ImportError:
        print("\n  ⚠ pyngrok não encontrado. Execute:")
        print("    pip install pyngrok")
        _run_local_only()

    except Exception as e:
        # ngrok failed (e.g., no auth token) — still serve locally
        print(f"\n  ⚠ ngrok não pôde ser iniciado: {e}")
        print("  Para URL pública, configure o token:")
        print("    ngrok config add-authtoken SEU_TOKEN")
        print("    (Crie conta gratuita em https://ngrok.com)")
        _run_local_only()


def _run_local_only():
    print(f"\n{'='*58}")
    print(f"  Servindo APENAS localmente")
    print(f"  URL: http://localhost:8000")
    print(f"{'='*58}")
    print("  Pressione Ctrl+C para encerrar.\n")

    try:
        webbrowser.open("http://localhost:8000")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Servidor encerrado.")


if __name__ == "__main__":
    main()
