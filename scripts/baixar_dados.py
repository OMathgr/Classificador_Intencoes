"""Baixa a base MASSIVE 1.1 (Amazon, CC BY 4.0) e extrai só o português (pt-PT).

Uso: python scripts/baixar_dados.py
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
ARQUIVOS = {
    "1.1/data/pt-PT.jsonl": "pt-PT.jsonl",
    "1.1/LICENSE": "MASSIVE_LICENSE",
    "1.1/NOTICE.md": "MASSIVE_NOTICE.md",
}
DESTINO = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        arquivo = Path(tmp) / "massive.tar.gz"
        print(f"Baixando {URL} (cerca de 40 MB)...")
        urllib.request.urlretrieve(URL, arquivo)
        with tarfile.open(arquivo) as tar:
            for membro, nome in ARQUIVOS.items():
                origem = tar.extractfile(membro)
                if origem is None:
                    raise SystemExit(f"Arquivo {membro} não encontrado no pacote.")
                with origem, open(DESTINO / nome, "wb") as saida:
                    shutil.copyfileobj(origem, saida)
    print(f"Pronto: dados em {DESTINO}")


if __name__ == "__main__":
    main()
