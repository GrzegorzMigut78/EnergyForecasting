# Dane

Surowe pliki danych nie są przechowywane w repozytorium (poufność, rozmiar).
Umieść je w `data/raw/` pod nazwami wskazanymi w `configs/datasets/*.yaml`:

| Plik | Zbiór | Skąd |
|---|---|---|
| `data/raw/ElecEner_long.xlsx` | ElecEner Long, 01.10.2001-30.06.2003 | od autorów artykułu |
| `data/raw/ElecEner_Short_d.xlsx` | dane pierwotnej wersji artykułu (tylko test zgodności z tabelą 1) | od autorów artykułu |
| `data/raw/gefcom2012/*.csv` | GEFCom2012, ścieżka obciążenia | `scripts/fetch_gefcom2012.py` |

Surowych plików nie należy edytować. Wszystkie korekty są opisane w pliku YAML zbioru
i wykonywane w kodzie. Skrót SHA-256 pliku zapisuje się przy każdym przebiegu
(`results/<przebieg>/environment.json`), więc wiadomo, na jakiej wersji danych liczono.

Kontrolny SHA-256 pliku ElecEner_long.xlsx użytego przy budowie projektu:
`9b6e08610b8937e4ae619cade349e296db8317577423fa471933275f2df12fa9`
